import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent.types import ExecutionTrace, ToolCallRecord, ToolPlan
from eval.replay import (
    build_run_id,
    build_diff,
    build_report,
    build_session_timelines,
    classify_case,
    compute_variant_summary,
    load_source_trace_records,
    run_replay,
    safe_session_key,
    summarize_trace,
    validate_sample_size,
    validate_input_root,
    validate_variants,
)


def make_trace(
    *,
    request_id: str,
    session_id: str,
    user_query: str = "question",
    intent: str = "kgqa",
    metadata: dict | None = None,
    failure_stage: str | None = None,
    failure_tags: list[str] | None = None,
    quality_signals: list[str] | None = None,
    plan: list | None = None,
    tool_calls: list | None = None,
    total_latency_ms: int = 0,
) -> ExecutionTrace:
    return ExecutionTrace(
        request_id=request_id,
        session_id=session_id,
        user_query=user_query,
        intent=intent,
        plan=plan or [],
        tool_calls=tool_calls or [],
        failure_stage=failure_stage,
        failure_tags=failure_tags or [],
        quality_signals=quality_signals or [],
        total_latency_ms=total_latency_ms,
        metadata=metadata or {},
    )


def write_trace(path: Path, trace: ExecutionTrace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = trace.model_dump() if hasattr(trace, "model_dump") else trace.dict()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class StubSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history = []


class StubStore:
    def __init__(self):
        self._session = None

    def get_or_create(self, session_id: str):
        self._session = StubSession(session_id)
        return self._session


class StubService:
    def __init__(self, variant_name: str, session_key: str, capture_root: Path):
        self.variant_name = variant_name
        self.session_key = session_key
        self.capture_root = capture_root
        self.calls = []
        self.closed = False

    def run_agent(
        self,
        question,
        *,
        history=None,
        session_id=None,
        align_entities=True,
        session=None,
        enable_session_memory=False,
    ):
        turn = len(self.calls) + 1
        history_snapshot = [dict(item) for item in (history or [])]
        self.calls.append(
            {
                "question": question,
                "history": history_snapshot,
                "session_id": session_id,
                "align_entities": align_entities,
                "enable_session_memory": enable_session_memory,
            }
        )
        metadata = {
            "aligned_entities": [{"param_name": "param_0", "entity": f"{question}_entity", "label": "Trademark"}],
            "executed_params": {"param_0": f"{question}_entity"} if align_entities else {"param_0": question},
            "cypher_query": f"MATCH (n) RETURN '{question}'",
            "query_result": [{"name": question}] if question != "q3" else [],
            "confirmed_entity_cache_updates": [],
            "session_failure_memory": None,
            "task_memory_hit_count": 1 if enable_session_memory and turn > 1 else 0,
            "task_memory_miss_count": 0 if enable_session_memory and turn > 1 else 1,
        }
        if enable_session_memory and turn > 1 and question != "q3":
            entity_call = ToolCallRecord(
                tool_name="entity_link_tool",
                input_payload={},
                ok=True,
                output_payload={},
                cache_hit=True,
                latency_ms=0,
            )
        else:
            entity_call = ToolCallRecord(
                tool_name="entity_link_tool",
                input_payload={},
                ok=True,
                output_payload={},
                cache_hit=False,
                latency_ms=1,
            )
        query_ok = question != "q3"
        graph_call = ToolCallRecord(
            tool_name="graph_query_tool",
            input_payload={},
            ok=query_ok,
            output_payload={"rows": metadata["query_result"]} if query_ok else {},
            metadata={} if query_ok else {"error": "empty"},
            latency_ms=2,
        )
        answer_call = ToolCallRecord(tool_name="answer_tool", input_payload={}, ok=True, output_payload={}, latency_ms=1)
        trace = make_trace(
            request_id=f"trc_20260517_12000{turn}_{self.variant_name[:8]:0<8}"[:28],
            session_id=session_id or f"{self.variant_name}_{self.session_key}",
            user_query=question,
            metadata=metadata,
            failure_stage="query" if not query_ok else None,
            failure_tags=["query_empty"] if not query_ok else [],
            quality_signals=[],
            plan=[
                ToolPlan(tool_name="entity_link_tool", arguments={"mode": "hybrid"}),
                ToolPlan(tool_name="graph_query_tool", arguments={"timeout_ms": 2000}),
                ToolPlan(tool_name="answer_tool", arguments={}),
            ],
            tool_calls=[entity_call, graph_call, answer_call],
            total_latency_ms=3,
        )
        return SimpleNamespace(answer=f"answer:{question}", session_id=trace.session_id, trace=trace)

    def build_session_commit_payload(self, result, *, user_message):
        return {
            "user_message": user_message,
            "assistant_message": result.answer,
            "trace_id": result.trace.request_id,
            "entity_cache_updates": [],
            "recent_failure": None,
        }

    def close(self):
        self.closed = True


class ReplayEvalTestCase(unittest.TestCase):
    def test_safe_session_key_hashes_unsafe_session_id(self):
        key = safe_session_key("session:/\\:*?unsafe")
        self.assertTrue(key.startswith("sess_"))
        self.assertNotIn("/", key)
        self.assertNotIn("\\", key)

    def test_validate_input_root_rejects_external_logs_traces_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            external = Path(tmp_dir) / "logs" / "traces"
            external.mkdir(parents=True, exist_ok=True)
            with self.assertRaisesRegex(ValueError, "logs/traces"):
                validate_input_root(external)

    def test_build_run_id_uses_microseconds_and_hash_suffix(self):
        run_id = build_run_id(Path(ROOT_DIR / "logs" / "traces"))
        self.assertRegex(run_id, r"^replay_\d{8}_\d{6}_\d{6}_[0-9a-f]{8}$")

    def test_loader_ignores_replay_capture_and_tool_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_root = Path(tmp_dir) / "logs" / "traces" / "2026-05-17"
            replay_root = Path(tmp_dir) / "logs" / "replay" / "run-1" / "trace_captures" / "current" / "sess_x" / "2026-05-17"
            write_trace(
                trace_root / "trc_20260517_010101_aaaaaaaa.json",
                make_trace(request_id="trc_20260517_010101_aaaaaaaa", session_id="s1"),
            )
            write_trace(
                trace_root / "trc_20260517_010102_bbbbbbbb.json",
                make_trace(
                    request_id="trc_20260517_010102_bbbbbbbb",
                    session_id="s1",
                    metadata={"tool_only": True},
                ),
            )
            write_trace(
                replay_root / "trc_20260517_010103_cccccccc.json",
                make_trace(request_id="trc_20260517_010103_cccccccc", session_id="s1"),
            )
            with patch("eval.replay.LOG_DIR", Path(tmp_dir) / "logs"):
                records = load_source_trace_records(Path(tmp_dir) / "logs" / "traces")
            self.assertEqual([record.trace.request_id for record in records], ["trc_20260517_010101_aaaaaaaa"])

    def test_loader_does_not_skip_legal_source_path_with_replay_like_tokens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_root = Path(tmp_dir) / "logs" / "traces" / "replay_notes" / "trace_captures_backup" / "2026-05-17"
            write_trace(
                trace_root / "trc_20260517_010101_aaaaaaaa.json",
                make_trace(request_id="trc_20260517_010101_aaaaaaaa", session_id="s1"),
            )
            with patch("eval.replay.LOG_DIR", Path(tmp_dir) / "logs"):
                records = load_source_trace_records(Path(tmp_dir) / "logs" / "traces")
            self.assertEqual([record.trace.request_id for record in records], ["trc_20260517_010101_aaaaaaaa"])

    def test_build_session_timelines_skips_same_second_ambiguous_session(self):
        records = [
            SimpleNamespace(
                trace=make_trace(request_id="trc_20260517_010101_aaaaaaaa", session_id="s1"),
                trace_date="2026-05-17",
                request_second="20260517_010101",
                request_sort_key=("2026-05-17", "20260517_010101", "trc_20260517_010101_aaaaaaaa"),
            ),
            SimpleNamespace(
                trace=make_trace(request_id="trc_20260517_010101_bbbbbbbb", session_id="s1"),
                trace_date="2026-05-17",
                request_second="20260517_010101",
                request_sort_key=("2026-05-17", "20260517_010101", "trc_20260517_010101_bbbbbbbb"),
            ),
        ]
        timelines, skipped = build_session_timelines(records)
        self.assertEqual(timelines, [])
        self.assertEqual(skipped["order_ambiguous"], ["s1"])

    def test_build_session_timelines_skips_missing_sort_keys(self):
        records = [
            SimpleNamespace(
                trace=make_trace(request_id="bad-request-id", session_id="s1"),
                trace_date="2026-05-17",
                request_second="",
                request_sort_key=("2026-05-17", "", "bad-request-id"),
            ),
            SimpleNamespace(
                trace=make_trace(request_id="trc_20260517_010101_aaaaaaaa", session_id="s2"),
                trace_date="",
                request_second="20260517_010101",
                request_sort_key=("", "20260517_010101", "trc_20260517_010101_aaaaaaaa"),
            ),
        ]
        timelines, skipped = build_session_timelines(records)
        self.assertEqual(timelines, [])
        self.assertEqual(skipped["missing_sort_key"], ["s1", "s2"])

    def test_summarize_trace_defaults_missing_fields(self):
        trace = make_trace(request_id="trc_20260517_010101_aaaaaaaa", session_id="s1")
        summary = summarize_trace(trace)
        self.assertEqual(summary["aligned_entities"], [])
        self.assertEqual(summary["executed_params"], {})
        self.assertEqual(summary["cypher_query"], "")
        self.assertEqual(summary["task_memory_hit_count"], 0)
        self.assertEqual(summary["task_memory_miss_count"], 0)
        self.assertFalse(summary["execution_success"])
        self.assertFalse(summary["non_empty_result"])

    def test_build_diff_normalizes_whitespace_and_dict_key_order(self):
        left = {
            "tool_plan": [{"tool_name": "entity_link_tool", "arguments": {"top_k": 3, "mode": "hybrid"}}],
            "aligned_entities": [{"label": "Trademark", "entity": "Apple", "param_name": "param_0"}],
            "executed_params": {"param_1": "B", "param_0": "A"},
            "cypher_query": " MATCH   (n)\nRETURN n ",
            "failure_stage": "",
            "failure_tags": [],
            "quality_signals": [],
            "execution_success": True,
            "non_empty_result": True,
            "total_latency_ms": 10,
            "task_memory_hit_count": 0,
            "task_memory_miss_count": 0,
            "entity_link_full_skip": False,
        }
        right = {
            **left,
            "tool_plan": [{"tool_name": "entity_link_tool", "arguments": {"mode": "hybrid", "top_k": 3}}],
            "aligned_entities": [{"param_name": "param_0", "entity": "Apple", "label": "Trademark"}],
            "executed_params": {"param_0": "A", "param_1": "B"},
            "cypher_query": "MATCH (n) RETURN n",
        }
        diff = build_diff(left, right)
        self.assertFalse(diff["tool_plan_changed"])
        self.assertFalse(diff["aligned_entities_changed"])
        self.assertFalse(diff["executed_params_changed"])
        self.assertFalse(diff["cypher_query_changed"])

    def test_classify_case_uses_non_empty_and_failure_stage_changes(self):
        source_summary = {
            "tool_plan": [],
            "aligned_entities": [],
            "executed_params": {},
            "cypher_query": "",
            "failure_stage": "plan",
            "failure_tags": ["plan_schema_invalid"],
            "quality_signals": [],
            "execution_success": False,
            "non_empty_result": False,
            "total_latency_ms": 0,
            "task_memory_hit_count": 0,
            "task_memory_miss_count": 0,
            "entity_link_full_skip": False,
        }
        replay_summary = {
            **source_summary,
            "failure_stage": "query",
            "execution_success": True,
            "non_empty_result": False,
        }
        diff = build_diff(source_summary, replay_summary)
        self.assertEqual(classify_case(source_summary, replay_summary, diff), "changed_same_outcome")

    def test_compute_variant_summary_handles_empty_records(self):
        summary = compute_variant_summary([])
        self.assertEqual(summary["hard_success_rate"], 0.0)
        self.assertEqual(summary["cache_hit_entity_ratio"], 0.0)
        self.assertIsNone(summary["latency_p50_ms"])
        self.assertIsNone(summary["latency_p95_ms"])

    def test_build_report_includes_missing_sort_key_stat(self):
        report = build_report(
            {
                "input_root": "logs/traces",
                "partial_session_view_assumed": False,
                "total_sessions": 0,
                "total_source_traces": 0,
                "skipped_sessions_order_ambiguous": 1,
                "skipped_sessions_missing_sort_key": 2,
                "warmup_turns_executed": 0,
                "warmup_sessions_executed": 0,
                "variants": {
                    "current": {
                        "total_target_traces": 0,
                        "hard_success_rate": 0.0,
                        "quality_clean_rate": 0.0,
                        "non_empty_result_rate": 0.0,
                        "cache_hit_entity_ratio": 0.0,
                        "cache_hit_turn_rate": 0.0,
                        "entity_link_full_skip_turn_rate": 0.0,
                        "latency_p50_ms": None,
                        "latency_p95_ms": None,
                        "improvement_count": 0,
                        "regression_count": 0,
                        "changed_same_outcome_count": 0,
                    }
                },
            },
            [],
        )
        self.assertIn("skipped_sessions_missing_sort_key", report)

    def test_validate_variants_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "Duplicate replay variants"):
            validate_variants(["current", "memory_on", "current"])

    def test_validate_sample_size_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_sample_size(-1)

    def test_run_replay_uses_session_sample_and_target_only_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_root = Path(tmp_dir) / "logs" / "traces"
            day = input_root / "2026-05-17"
            write_trace(day / "trc_20260517_010101_aaaaaaaa.json", make_trace(request_id="trc_20260517_010101_aaaaaaaa", session_id="s1", user_query="q1", failure_tags=["query_empty"], failure_stage="query"))
            write_trace(day / "trc_20260517_010102_bbbbbbbb.json", make_trace(request_id="trc_20260517_010102_bbbbbbbb", session_id="s1", user_query="q2", failure_tags=["query_empty"], failure_stage="query"))
            write_trace(day / "trc_20260517_010103_cccccccc.json", make_trace(request_id="trc_20260517_010103_cccccccc", session_id="s2", user_query="q3", failure_tags=["query_empty"], failure_stage="query"))

            def service_factory(variant_name, capture_root, session_key):
                return SimpleNamespace(
                    service=StubService(variant_name, session_key, capture_root),
                    runtime=None,
                    close=lambda: None,
                )

            class Bundle:
                def __init__(self, variant_name, capture_root, session_key):
                    self.service = StubService(variant_name, session_key, capture_root)
                    self.runtime = None

                def close(self):
                    self.service.close()

            with patch("eval.replay.LOG_DIR", Path(tmp_dir) / "logs"):
                summary, rows = run_replay(
                    input_root=input_root,
                    variants=["current", "memory_on"],
                    sample_size=1,
                    failure_tag="query_empty",
                    output_root=Path(tmp_dir) / "logs" / "replay" / "run-1",
                    service_factory=lambda variant_name, capture_root, session_key: Bundle(variant_name, capture_root, session_key),
                )

            self.assertEqual(summary["total_sessions"], 1)
            self.assertEqual(summary["total_source_traces"], 2)
            self.assertEqual(len(rows), 4)
            self.assertEqual(summary["warmup_turns_executed"], 0)
            self.assertIn("current", summary["variants"])
            self.assertIn("memory_on", summary["variants"])
            manifest_path = Path(tmp_dir) / "logs" / "replay" / "run-1" / "session_manifest.jsonl"
            manifest_rows = manifest_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(manifest_rows), 2)


if __name__ == "__main__":
    unittest.main()
