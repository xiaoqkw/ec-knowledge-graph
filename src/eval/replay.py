from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import AgentController, AgentResources, AgentRuntime
from agent.runtime import TraceStore, dump_model, parse_model
from agent.types import ExecutionTrace, ToolCallRecord
from configuration.config import LOG_DIR
from eval.kgqa_eval import _commit_eval_session_turn
from web.memory import InMemoryQASessionStore
from web.service import ChatService


REQUEST_ID_PATTERN = re.compile(r"^trc_(\d{8}_\d{6})_[0-9a-f]{8}$")
DATE_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
WHITESPACE_PATTERN = re.compile(r"\s+")
REPLAY_VARIANTS = {
    "current": {"align_entities": True, "enable_session_memory": False},
    "memory_on": {"align_entities": True, "enable_session_memory": True},
    "ablation": {"align_entities": False, "enable_session_memory": False},
}


@dataclass
class TraceRecord:
    path: Path
    trace: ExecutionTrace
    trace_date: str
    request_second: str
    request_sort_key: tuple[str, str, str]


@dataclass
class SessionTimeline:
    session_id: str
    records: list[TraceRecord]
    order_ambiguous: bool
    first_sort_key: tuple[str, str, str]


@dataclass
class ReplayServiceBundle:
    service: Any
    runtime: Any = None

    def close(self) -> None:
        try:
            close = getattr(self.service, "close", None)
            if callable(close):
                close()
        finally:
            if self.runtime is not None:
                self.runtime.close()


def safe_session_key(session_id: str) -> str:
    digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:16]
    return f"sess_{digest}"


def validate_input_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed_root = (LOG_DIR / "traces").resolve()
    if resolved == allowed_root or allowed_root in resolved.parents:
        return resolved
    raise ValueError("Replay source must point to logs/traces or one of its subdirectories.")


def is_replay_capture_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    for index in range(len(parts) - 3):
        if parts[index] != "logs":
            continue
        if parts[index + 1] != "replay":
            continue
        if parts[index + 3] != "trace_captures":
            continue
        return True
    return False


def partial_session_view_assumed(input_root: Path) -> bool:
    return input_root.name.lower() != "traces"


def extract_trace_date(path: Path) -> str:
    for part in reversed(path.parts):
        if DATE_DIR_PATTERN.match(part):
            return part
    return ""


def extract_request_second(request_id: str) -> str:
    matched = REQUEST_ID_PATTERN.match(request_id or "")
    if matched is None:
        return ""
    return matched.group(1)


def load_source_trace_records(input_root: Path) -> list[TraceRecord]:
    source_root = validate_input_root(input_root)
    records = []
    for path in sorted(source_root.rglob("*.json")):
        if is_replay_capture_path(path):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        trace = parse_model(ExecutionTrace, payload)
        if trace.intent != "kgqa":
            continue
        if bool(trace.metadata.get("tool_only", False)):
            continue
        trace_date = extract_trace_date(path)
        request_second = extract_request_second(trace.request_id)
        records.append(
            TraceRecord(
                path=path,
                trace=trace,
                trace_date=trace_date,
                request_second=request_second,
                request_sort_key=(trace_date, request_second, trace.request_id),
            )
        )
    return records


def build_session_timelines(records: list[TraceRecord]) -> tuple[list[SessionTimeline], dict[str, list[str]]]:
    grouped: dict[str, list[TraceRecord]] = {}
    for record in records:
        grouped.setdefault(record.trace.session_id, []).append(record)

    timelines = []
    skipped = {"order_ambiguous": [], "missing_sort_key": []}
    for session_id, session_records in grouped.items():
        ordered = sorted(session_records, key=lambda item: item.request_sort_key)
        if any(not record.trace_date or not record.request_second for record in ordered):
            skipped["missing_sort_key"].append(session_id)
            continue
        seen_keys = set()
        ambiguous = False
        for record in ordered:
            second_key = (record.trace_date, record.request_second)
            if second_key in seen_keys:
                ambiguous = True
                break
            seen_keys.add(second_key)
        if ambiguous:
            skipped["order_ambiguous"].append(session_id)
            continue
        first_sort_key = ordered[0].request_sort_key if ordered else ("", "", session_id)
        timelines.append(
            SessionTimeline(
                session_id=session_id,
                records=ordered,
                order_ambiguous=False,
                first_sort_key=first_sort_key,
            )
        )
    timelines.sort(key=lambda item: (item.first_sort_key[0], item.first_sort_key[1], item.session_id))
    skipped["order_ambiguous"].sort()
    skipped["missing_sort_key"].sort()
    return timelines, skipped


def select_timelines(
    timelines: list[SessionTimeline],
    *,
    session_ids: set[str] | None = None,
    failure_tag: str | None = None,
    quality_signal: str | None = None,
    sample_size: int | None = None,
) -> tuple[list[SessionTimeline], set[str]]:
    matching_ids: set[str] = set()
    candidate_timelines: list[SessionTimeline] = []
    for timeline in timelines:
        session_matches = []
        for record in timeline.records:
            trace = record.trace
            if session_ids and trace.session_id not in session_ids:
                continue
            if failure_tag and failure_tag not in trace.failure_tags:
                continue
            if quality_signal and quality_signal not in trace.quality_signals:
                continue
            session_matches.append(trace.request_id)
        if session_matches:
            candidate_timelines.append(timeline)
            matching_ids.update(session_matches)

    if sample_size is not None:
        candidate_timelines = candidate_timelines[:sample_size]
        allowed_sessions = {timeline.session_id for timeline in candidate_timelines}
        matching_ids = {
            trace_id
            for trace_id in matching_ids
            if any(record.trace.request_id == trace_id and record.trace.session_id in allowed_sessions for timeline in candidate_timelines for record in timeline.records)
        }

    return candidate_timelines, matching_ids


def validate_variants(variants: list[str]) -> list[str]:
    if not variants:
        return ["current"]
    duplicated = []
    seen = set()
    for variant in variants:
        if variant in seen and variant not in duplicated:
            duplicated.append(variant)
        seen.add(variant)
    if duplicated:
        raise ValueError(f"Duplicate replay variants are not allowed: {', '.join(duplicated)}")
    unknown = [variant for variant in variants if variant not in REPLAY_VARIANTS]
    if unknown:
        raise ValueError(f"Unsupported replay variants: {', '.join(unknown)}")
    return variants


def validate_sample_size(sample_size: int | None) -> int | None:
    if sample_size is None:
        return None
    if sample_size < 0:
        raise ValueError("--sample must be a non-negative integer.")
    return sample_size


def build_run_id(input_root: Path) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = hashlib.sha1(str(input_root.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"replay_{timestamp}_{suffix}"


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", str(text or "").strip())


def stable_serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_tool_plan(plan: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for item in plan or []:
        payload = dump_model(item)
        if not isinstance(payload, dict):
            continue
        normalized.append(
            {
                "tool_name": str(payload.get("tool_name", "")),
                "arguments": payload.get("arguments", {}) if isinstance(payload.get("arguments"), dict) else {},
            }
        )
    return normalized


def normalize_aligned_entities(items: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for item in items or []:
        payload = dump_model(item)
        if isinstance(payload, dict):
            normalized.append(payload)
    return normalized


def trace_execution_success(trace: ExecutionTrace) -> bool:
    for call in trace.tool_calls:
        if call.tool_name == "graph_query_tool" and call.ok:
            return True
    return False


def trace_non_empty_result(trace: ExecutionTrace) -> bool:
    return bool(trace.metadata.get("query_result", []))


def trace_entity_link_full_skip(trace: ExecutionTrace) -> bool:
    for call in trace.tool_calls:
        if call.tool_name == "entity_link_tool" and getattr(call, "cache_hit", False):
            return True
    return False


def summarize_trace(trace: ExecutionTrace) -> dict[str, Any]:
    metadata = trace.metadata or {}
    return {
        "tool_plan": normalize_tool_plan(trace.plan),
        "aligned_entities": normalize_aligned_entities(metadata.get("aligned_entities", [])),
        "executed_params": metadata.get("executed_params", {}) if isinstance(metadata.get("executed_params"), dict) else {},
        "cypher_query": str(metadata.get("cypher_query", "")),
        "failure_stage": str(trace.failure_stage or ""),
        "failure_tags": list(trace.failure_tags or []),
        "quality_signals": list(trace.quality_signals or []),
        "execution_success": trace_execution_success(trace),
        "non_empty_result": trace_non_empty_result(trace),
        "total_latency_ms": int(getattr(trace, "total_latency_ms", 0) or 0),
        "task_memory_hit_count": int(metadata.get("task_memory_hit_count", 0) or 0),
        "task_memory_miss_count": int(metadata.get("task_memory_miss_count", 0) or 0),
        "entity_link_full_skip": trace_entity_link_full_skip(trace),
    }


def canonical_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_plan": [stable_serialize(item) for item in summary["tool_plan"]],
        "aligned_entities": [stable_serialize(item) for item in summary["aligned_entities"]],
        "executed_params": stable_serialize(summary["executed_params"]),
        "cypher_query": normalize_whitespace(summary["cypher_query"]),
        "failure_stage": normalize_whitespace(summary["failure_stage"]),
        "failure_tags": sorted(str(item) for item in summary["failure_tags"]),
        "quality_signals": sorted(str(item) for item in summary["quality_signals"]),
        "execution_success": bool(summary["execution_success"]),
        "non_empty_result": bool(summary["non_empty_result"]),
    }


def build_diff(source_summary: dict[str, Any], replay_summary: dict[str, Any]) -> dict[str, Any]:
    left = canonical_summary(source_summary)
    right = canonical_summary(replay_summary)
    return {
        "tool_plan_changed": left["tool_plan"] != right["tool_plan"],
        "aligned_entities_changed": left["aligned_entities"] != right["aligned_entities"],
        "executed_params_changed": left["executed_params"] != right["executed_params"],
        "cypher_query_changed": left["cypher_query"] != right["cypher_query"],
        "failure_stage_changed": left["failure_stage"] != right["failure_stage"],
        "failure_tags_changed": left["failure_tags"] != right["failure_tags"],
        "quality_signals_changed": left["quality_signals"] != right["quality_signals"],
        "execution_success_changed": left["execution_success"] != right["execution_success"],
        "non_empty_result_changed": left["non_empty_result"] != right["non_empty_result"],
        "latency_total_delta_ms": replay_summary["total_latency_ms"] - source_summary["total_latency_ms"],
    }


def is_hard_success(summary: dict[str, Any]) -> bool:
    return bool(summary["execution_success"] and summary["non_empty_result"] and not summary["failure_tags"])


def is_quality_clean(summary: dict[str, Any]) -> bool:
    return bool(is_hard_success(summary) and not summary["quality_signals"])


def classify_case(source_summary: dict[str, Any], replay_summary: dict[str, Any], diff: dict[str, Any]) -> str:
    source_success = is_hard_success(source_summary)
    replay_success = is_hard_success(replay_summary)
    if not source_success and replay_success:
        return "improvement"
    if source_success and not replay_success:
        return "regression"
    comparison_keys = (
        "tool_plan_changed",
        "aligned_entities_changed",
        "executed_params_changed",
        "cypher_query_changed",
        "failure_stage_changed",
        "failure_tags_changed",
        "quality_signals_changed",
        "execution_success_changed",
        "non_empty_result_changed",
    )
    if any(diff[key] for key in comparison_keys):
        return "changed_same_outcome"
    return "unchanged"


def percentile(values: list[int], ratio: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(math.ceil(len(ordered) * ratio) - 1, 0)
    return ordered[index]


def rate(values: list[bool]) -> float:
    return sum(1 for item in values if item) / len(values) if values else 0.0


def create_live_bundle(variant_name: str, capture_root: Path, session_key: str) -> ReplayServiceBundle:
    variant_capture_root = capture_root / variant_name / session_key
    runtime = AgentRuntime(resources=AgentResources(), trace_store=TraceStore(variant_capture_root))
    controller = AgentController(runtime, enable_explain=True)
    service = ChatService(controller=controller)
    return ReplayServiceBundle(service=service, runtime=runtime)


def replay_session_timeline(
    timeline: SessionTimeline,
    *,
    variant_name: str,
    selected_trace_ids: set[str],
    capture_root: Path,
    partial_view_assumed: bool,
    service_factory: Callable[[str, Path, str], ReplayServiceBundle] = create_live_bundle,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    session_key = safe_session_key(timeline.session_id)
    bundle = service_factory(variant_name, capture_root, session_key)
    service = bundle.service
    store = InMemoryQASessionStore()
    replay_session = store.get_or_create(f"replay_{variant_name}_{session_key}")
    variant_config = REPLAY_VARIANTS[variant_name]

    records: list[dict[str, Any]] = []
    stats = {"warmup_turns_executed": 0, "warmup_sessions_executed": 0}
    had_warmup = False
    try:
        for turn_index, source_record in enumerate(timeline.records, start=1):
            visible_history_count = len(replay_session.history)
            result = service.run_agent(
                source_record.trace.user_query,
                history=replay_session.history,
                session_id=replay_session.session_id,
                align_entities=variant_config["align_entities"],
                session=replay_session,
                enable_session_memory=variant_config["enable_session_memory"],
            )
            _commit_eval_session_turn(store, replay_session, service, result, user_message=source_record.trace.user_query)
            if source_record.trace.request_id not in selected_trace_ids:
                stats["warmup_turns_executed"] += 1
                had_warmup = True
                continue
            source_summary = summarize_trace(source_record.trace)
            replay_summary = summarize_trace(result.trace)
            diff = build_diff(source_summary, replay_summary)
            records.append(
                {
                    "variant": variant_name,
                    "source_trace_id": source_record.trace.request_id,
                    "source_session_id": timeline.session_id,
                    "source_turn_index_in_visible_timeline": turn_index,
                    "visible_history_count": visible_history_count,
                    "partial_session_view_assumed": partial_view_assumed,
                    "source_summary": source_summary,
                    "replay_summary": replay_summary,
                    "diff": diff,
                    "classification": classify_case(source_summary, replay_summary, diff),
                }
            )
        if had_warmup:
            stats["warmup_sessions_executed"] = 1
        return records, stats
    finally:
        bundle.close()


def compute_variant_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    replay_summaries = [record["replay_summary"] for record in records]
    latencies = [summary["total_latency_ms"] for summary in replay_summaries]
    hit_sum = sum(summary["task_memory_hit_count"] for summary in replay_summaries)
    miss_sum = sum(summary["task_memory_miss_count"] for summary in replay_summaries)
    denominator = hit_sum + miss_sum
    return {
        "total_target_traces": len(records),
        "hard_success_rate": rate([is_hard_success(summary) for summary in replay_summaries]),
        "quality_clean_rate": rate([is_quality_clean(summary) for summary in replay_summaries]),
        "non_empty_result_rate": rate([bool(summary["non_empty_result"]) for summary in replay_summaries]),
        "cache_hit_entity_ratio": hit_sum / denominator if denominator else 0.0,
        "cache_hit_turn_rate": rate([summary["task_memory_hit_count"] > 0 for summary in replay_summaries]),
        "entity_link_full_skip_turn_rate": rate([bool(summary["entity_link_full_skip"]) for summary in replay_summaries]),
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "improvement_count": sum(1 for record in records if record["classification"] == "improvement"),
        "regression_count": sum(1 for record in records if record["classification"] == "regression"),
        "changed_same_outcome_count": sum(1 for record in records if record["classification"] == "changed_same_outcome"),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_report(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Replay Summary",
        "",
        f"- `input_root`: `{summary['input_root']}`",
        f"- `partial_session_view_assumed`: `{summary['partial_session_view_assumed']}`",
        f"- `total_sessions`: `{summary['total_sessions']}`",
        f"- `total_source_traces`: `{summary['total_source_traces']}`",
        f"- `skipped_sessions_order_ambiguous`: `{summary['skipped_sessions_order_ambiguous']}`",
        f"- `skipped_sessions_missing_sort_key`: `{summary['skipped_sessions_missing_sort_key']}`",
        f"- `warmup_turns_executed`: `{summary['warmup_turns_executed']}`",
        f"- `warmup_sessions_executed`: `{summary['warmup_sessions_executed']}`",
        "",
        "## Variant Summary",
        "",
        "| variant | targets | hard_success | quality_clean | non_empty | cache_entity | cache_turn | full_skip_turn | p50 | p95 | improve | regress | changed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, metrics in summary["variants"].items():
        lines.append(
            "| "
            f"{variant} | {metrics['total_target_traces']} | {metrics['hard_success_rate']:.4f} | "
            f"{metrics['quality_clean_rate']:.4f} | {metrics['non_empty_result_rate']:.4f} | "
            f"{metrics['cache_hit_entity_ratio']:.4f} | {metrics['cache_hit_turn_rate']:.4f} | "
            f"{metrics['entity_link_full_skip_turn_rate']:.4f} | "
            f"{metrics['latency_p50_ms'] if metrics['latency_p50_ms'] is not None else 'null'} | "
            f"{metrics['latency_p95_ms'] if metrics['latency_p95_ms'] is not None else 'null'} | "
            f"{metrics['improvement_count']} | {metrics['regression_count']} | "
            f"{metrics['changed_same_outcome_count']} |"
        )

    for title, classification in (("Regression Cases", "regression"), ("Improvement Cases", "improvement")):
        lines.extend(["", f"## {title}", ""])
        matched = [record for record in records if record["classification"] == classification]
        if not matched:
            lines.append("- none")
            continue
        for record in matched:
            lines.append(
                "- "
                f"`{record['variant']}` / `{record['source_trace_id']}` / "
                f"`visible_history_count={record['visible_history_count']}`"
            )
    return "\n".join(lines) + "\n"


def build_session_manifest_rows(variant_name: str, timelines: list[SessionTimeline], selected_trace_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for timeline in timelines:
        source_trace_ids = [record.trace.request_id for record in timeline.records if record.trace.request_id in selected_trace_ids]
        if not source_trace_ids:
            continue
        rows.append(
            {
                "variant": variant_name,
                "session_key": safe_session_key(timeline.session_id),
                "source_session_id": timeline.session_id,
                "source_trace_ids": source_trace_ids,
            }
        )
    return rows


def run_replay(
    *,
    input_root: Path,
    variants: list[str],
    sample_size: int | None = None,
    session_ids: set[str] | None = None,
    failure_tag: str | None = None,
    quality_signal: str | None = None,
    output_root: Path | None = None,
    service_factory: Callable[[str, Path, str], ReplayServiceBundle] = create_live_bundle,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variants = validate_variants(list(variants))
    sample_size = validate_sample_size(sample_size)
    records = load_source_trace_records(input_root)
    timelines, skipped = build_session_timelines(records)
    selected_timelines, selected_trace_ids = select_timelines(
        timelines,
        session_ids=session_ids,
        failure_tag=failure_tag,
        quality_signal=quality_signal,
        sample_size=sample_size,
    )
    partial_view = partial_session_view_assumed(validate_input_root(input_root))
    run_id = build_run_id(input_root)
    if output_root is None:
        output_root = LOG_DIR / "replay" / run_id

    all_rows: list[dict[str, Any]] = []
    warmup_turns = 0
    warmup_sessions = 0
    manifests = []
    variants_summary = {}

    for variant_name in variants:
        variant_rows: list[dict[str, Any]] = []
        manifests.extend(build_session_manifest_rows(variant_name, selected_timelines, selected_trace_ids))
        for timeline in selected_timelines:
            session_rows, stats = replay_session_timeline(
                timeline,
                variant_name=variant_name,
                selected_trace_ids=selected_trace_ids,
                capture_root=output_root / "trace_captures",
                partial_view_assumed=partial_view,
                service_factory=service_factory,
            )
            warmup_turns += stats["warmup_turns_executed"]
            warmup_sessions += stats["warmup_sessions_executed"]
            variant_rows.extend(session_rows)
        variants_summary[variant_name] = compute_variant_summary(variant_rows)
        all_rows.extend(variant_rows)

    summary = {
        "input_root": str(validate_input_root(input_root)),
        "partial_session_view_assumed": partial_view,
        "total_sessions": len(selected_timelines),
        "total_source_traces": len(selected_trace_ids),
        "skipped_sessions_order_ambiguous": len(skipped["order_ambiguous"]),
        "skipped_sessions_missing_sort_key": len(skipped["missing_sort_key"]),
        "skipped_session_ids_order_ambiguous": skipped["order_ambiguous"],
        "skipped_session_ids_missing_sort_key": skipped["missing_sort_key"],
        "warmup_turns_executed": warmup_turns,
        "warmup_sessions_executed": warmup_sessions,
        "variants": variants_summary,
    }

    write_jsonl(output_root / "replay_results.jsonl", all_rows)
    write_json(output_root / "replay_summary.json", summary)
    write_jsonl(output_root / "session_manifest.jsonl", manifests)
    (output_root / "replay_summary.md").write_text(build_report(summary, all_rows), encoding="utf-8")
    return summary, all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay historical KGQA traces against current runtime variants.")
    parser.add_argument("--traces", required=True, help="Path under logs/traces or one of its subdirectories.")
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(REPLAY_VARIANTS),
        help="Replay variant to run. Can be specified multiple times. Defaults to current.",
    )
    parser.add_argument("--sample", type=int, default=None, help="Sample sessions deterministically by first visible trace ordering.")
    parser.add_argument("--session-id", action="append", default=None, help="Restrict replay to matching source session_id values.")
    parser.add_argument("--failure-tag", default=None, help="Only report source traces containing this failure tag.")
    parser.add_argument("--quality-signal", default=None, help="Only report source traces containing this quality signal.")
    parser.add_argument("--output-root", default=None, help="Optional override for logs/replay/<run_id> output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root) if args.output_root else None
    try:
        run_replay(
            input_root=Path(args.traces),
            variants=args.variant or ["current"],
            sample_size=args.sample,
            session_ids=set(args.session_id or []),
            failure_tag=args.failure_tag,
            quality_signal=args.quality_signal,
            output_root=output_root,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "stage": "replay_execute",
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
