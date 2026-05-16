import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import AgentController, AgentResources, AgentRuntime
from configuration.config import LOG_DIR
from dialogue.service import DialogueService
from dialogue.types import RecommendationItem

PRIMARY_FAILURE_ORDER = ("routing", "compare", "state", "tool_invocation_mismatch", "retrieval")


def load_dataset(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_item(payload: dict) -> RecommendationItem:
    return RecommendationItem(
        sku_id=int(payload["sku_id"]),
        spu_id=int(payload["spu_id"]),
        sku_name=payload["sku_name"],
        spu_name=payload["spu_name"],
        brand=payload["brand"],
        price=float(payload["price"]),
        reason=payload.get("reason", "fixture"),
        default_img=payload.get("default_img", ""),
        storage_options=list(payload.get("storage_options", [])),
        source_text=payload.get("source_text", payload["sku_name"]),
    )


class StubNLU:
    def __init__(self, outputs: list[dict]):
        self.outputs = list(outputs)

    def parse(self, message, *, brand_vocabulary, state_has_context):
        payload = self.outputs.pop(0)
        return type("NLUResult", (), payload)


class FixtureRetriever:
    def __init__(self, fixture: dict):
        self.fixture = fixture

    def load_brand_vocabulary(self):
        return ["苹果", "华为", "小米", "OPPO", "VIVO", "红米", "Apple"]

    def load_storage_vocabulary(self):
        return ["128G", "256G", "512G"]

    def search(self, slots, limit=3):
        normalized = tuple(sorted((key, value) for key, value in slots.items() if value is not None))
        for candidate in self.fixture.get("search_cases", []):
            candidate_key = tuple(sorted(candidate.get("slots", {}).items()))
            if candidate_key == normalized:
                return [build_item(item) for item in candidate.get("results", [])[:limit]]
        return []

    def compare(self, spu_ids, use_case):
        return self.fixture.get("compare_text", "comparison")

    def get_min_price(self, brand=None):
        return self.fixture.get("min_price_by_brand", {}).get(brand, self.fixture.get("min_price"))

    def close(self):
        return None


class RecordingRuntime(AgentRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._turn_tool_calls: list[str] = []

    def begin_turn(self) -> None:
        self._turn_tool_calls = []

    def consume_turn_tool_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tool_name in self._turn_tool_calls:
            counts[tool_name] = counts.get(tool_name, 0) + 1
        self._turn_tool_calls = []
        return counts

    def run_tool_only(
        self,
        *,
        tool_name: str,
        payload: dict,
        user_query: str,
        session_id: str | None = None,
        intent: str = "dialogue_tool",
    ):
        self._turn_tool_calls.append(tool_name)
        return super().run_tool_only(
            tool_name=tool_name,
            payload=payload,
            user_query=user_query,
            session_id=session_id,
            intent=intent,
        )


def evaluate_state_transition(turn: dict, response: dict) -> list[str]:
    evidence = []
    state = response["state"]
    if turn.get("expect_mode") and response["mode"] != turn["expect_mode"]:
        evidence.append("routing")
    if turn.get("expect_action") and response["action"] != turn["expect_action"]:
        if turn.get("expect_action") == "compare":
            evidence.append("compare")
        else:
            evidence.append("state")
    for key, value in turn.get("expect_filled_slots", {}).items():
        if state["filled_slots"].get(key) != value:
            evidence.append("state")
            break
    if "expect_pending_slots" in turn and state.get("pending_slots", []) != turn["expect_pending_slots"]:
        evidence.append("state")
    if turn.get("expect_suggested_budget_min") is not None and state.get("suggested_budget_min") != turn["expect_suggested_budget_min"]:
        evidence.append("state")
    if turn.get("expect_reset") and state["filled_slots"]:
        evidence.append("routing")
    return evidence


def evaluate_tool_invocation(turn: dict, actual_tool_counts: dict[str, int]) -> list[str]:
    expected_tool_counts = dict(turn.get("expected_tool_counts", {}))
    if expected_tool_counts == dict(actual_tool_counts):
        return []
    return ["tool_invocation_mismatch"]


def evaluate_retrieval_outcome(turn: dict, response: dict) -> list[str]:
    recommendations = response["recommendations"]
    evidence = []
    if turn.get("min_recommendations", 0) > len(recommendations):
        evidence.append("retrieval")
    if turn.get("expected_brands_in_top3"):
        brands = {item["brand"] for item in recommendations[:3]}
        if not any(brand in brands for brand in turn["expected_brands_in_top3"]):
            evidence.append("retrieval")
    return evidence


def _primary_cause(evidence: list[str]) -> str | None:
    for label in PRIMARY_FAILURE_ORDER:
        if label in evidence:
            return label
    return evidence[0] if evidence else None


def evaluate_turn(task: dict, turn: dict, response: dict, actual_tool_counts: dict[str, int] | None = None) -> tuple[list[str], str | None]:
    state_evidence = evaluate_state_transition(turn, response)
    tool_evidence = evaluate_tool_invocation(turn, actual_tool_counts or {})
    retrieval_evidence = evaluate_retrieval_outcome(turn, response)
    evidence = state_evidence + tool_evidence + retrieval_evidence
    return evidence, _primary_cause(evidence)


def _is_runtime_success_turn(turn: dict, response: dict, evidence: list[str]) -> bool:
    if evidence:
        return False
    terminal_actions = {"recommend", "compare", "fallback_qa", "reset"}
    expected_action = turn.get("expect_action")
    if expected_action not in terminal_actions:
        return False
    return response["action"] == expected_action


def compute_summary(records: list[dict], tasks: list[dict]) -> dict:
    total_tasks = len(tasks)
    successful_tasks = 0
    turns_to_success = []
    state_transition_hits = 0
    tool_invocation_hits = 0
    total_turns = len(records)
    recommendation_turns = 0
    recommendation_hits = 0
    compare_turns = 0
    compare_hits = 0
    fallback_turns = 0
    failure_breakdown = {}

    task_status = {task["task_id"]: True for task in tasks}
    task_success_turn = {}
    task_last_turn = {}

    for record in records:
        task_last_turn[record["task_id"]] = max(task_last_turn.get(record["task_id"], 0), record["turn_index"])
        if record["state_transition_passed"]:
            state_transition_hits += 1
        if record["tool_invocation_passed"]:
            tool_invocation_hits += 1
        if record["passed"]:
            task_id = record["task_id"]
            if task_success_turn.get(task_id) is None and record.get("success_turn_reached"):
                task_success_turn[task_id] = record["turn_index"]
        else:
            task_status[record["task_id"]] = False
            primary = record.get("primary_cause") or "unknown"
            failure_breakdown[primary] = failure_breakdown.get(primary, 0) + 1
        if record.get("expected_action") == "recommend":
            recommendation_turns += 1
            if record["recommendation_count"] > 0:
                recommendation_hits += 1
        if record.get("expected_action") == "compare":
            compare_turns += 1
            if record["action"] == "compare" and record["state_transition_passed"]:
                compare_hits += 1
        if record["mode"] == "qa_fallback":
            fallback_turns += 1

    for task in tasks:
        success_turn = task_success_turn.get(task["task_id"])
        if task_status[task["task_id"]]:
            successful_tasks += 1
            turns_to_success.append(success_turn if success_turn is not None else task_last_turn.get(task["task_id"], 0))

    return {
        "total_tasks": total_tasks,
        "task_success_rate": successful_tasks / total_tasks if total_tasks else 0.0,
        "state_transition_correct_rate": state_transition_hits / total_turns if total_turns else 0.0,
        "tool_invocation_correct_rate": tool_invocation_hits / total_turns if total_turns else 0.0,
        "compare_success_rate": compare_hits / compare_turns if compare_turns else 0.0,
        "avg_turns_to_success": sum(turns_to_success) / len(turns_to_success) if turns_to_success else 0.0,
        "fallback_rate": fallback_turns / total_turns if total_turns else 0.0,
        "recommendation_non_empty_rate": recommendation_hits / recommendation_turns if recommendation_turns else 0.0,
        "failure_breakdown_primary": failure_breakdown,
    }


def write_logs(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def print_report(summary: dict) -> None:
    print("| metric | value |")
    print("| --- | ---: |")
    for key, value in summary.items():
        if isinstance(value, dict):
            continue
        print(f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dialogue state-machine/tool-invocation tasks with stub NLU.")
    parser.add_argument("--dataset", default=str(ROOT_DIR / "data" / "eval" / "dialogue_tasks.jsonl"))
    args = parser.parse_args()

    tasks = load_dataset(Path(args.dataset))
    records = []
    for task in tasks:
        retriever = FixtureRetriever(task["fixture"])
        runtime = RecordingRuntime(resources=AgentResources(llm_enabled=False))
        controller = AgentController(runtime, retriever=retriever, enable_explain=False)
        service = DialogueService(
            nlu=StubNLU(task["nlu_outputs"]),
            retriever=retriever,
            llm_enabled=False,
            agent_controller=controller,
        )
        try:
            session_id = None
            for index, turn in enumerate(task["turns"], start=1):
                runtime.begin_turn()
                response = service.chat(turn["user"], session_id=session_id, qa_handler=lambda _: "qa fallback")
                session_id = response["session_id"]
                actual_tool_counts = runtime.consume_turn_tool_counts()
                state_transition_passed = not evaluate_state_transition(turn, response)
                tool_invocation_passed = not evaluate_tool_invocation(turn, actual_tool_counts)
                evidence, primary = evaluate_turn(task, turn, response, actual_tool_counts)
                record = {
                    "task_id": task["task_id"],
                    "category": task.get("category"),
                    "turn_index": index,
                    "user_message": turn["user"],
                    "mode": response["mode"],
                    "action": response["action"],
                    "filled_slots": response["state"]["filled_slots"],
                    "pending_slots": response["state"]["pending_slots"],
                    "suggested_budget_min": response["state"].get("suggested_budget_min"),
                    "recommendation_count": len(response["recommendations"]),
                    "expected_action": turn.get("expect_action"),
                    "expected_filled_slots": turn.get("expect_filled_slots", {}),
                    "expected_suggested_budget_min": turn.get("expect_suggested_budget_min"),
                    "expected_tool_counts": dict(turn.get("expected_tool_counts", {})),
                    "actual_tool_counts": actual_tool_counts,
                    "state_transition_passed": state_transition_passed,
                    "tool_invocation_passed": tool_invocation_passed,
                    "success_turn_reached": _is_runtime_success_turn(turn, response, evidence),
                    "failure_evidence": evidence,
                    "primary_cause": primary,
                    "passed": not evidence,
                }
                records.append(record)
        finally:
            service.close()

    summary = compute_summary(records, tasks)
    write_logs(LOG_DIR / "eval" / "dialogue_offline.jsonl", records)
    write_summary(LOG_DIR / "eval" / "dialogue_summary.json", summary)
    print_report(summary)


if __name__ == "__main__":
    main()
