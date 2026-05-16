import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import AgentController, AgentResources, AgentRuntime
from configuration.config import DEEPSEEK_API_KEY
from dialogue.retrieval import PhoneGuideRetriever
from dialogue.service import DialogueService
from web.memory import CachedAlignedEntity, FailureMemory, InMemoryQASessionStore
from web.service import ChatService


def load_dataset(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoke dialogue cases against real Neo4j and real LLM.")
    parser.add_argument("--dataset", default=str(ROOT_DIR / "data" / "eval" / "dialogue_smoke_cases.jsonl"))
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        print(
            json.dumps(
                {
                    "status": "error",
                    "stage": "smoke_init",
                    "message": "DEEPSEEK_API_KEY is required for dialogue smoke evaluation.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1)

    try:
        retriever = PhoneGuideRetriever()
        runtime = AgentRuntime(resources=AgentResources())
        controller = AgentController(runtime, retriever=retriever, enable_explain=False)
        dialogue_service = DialogueService(retriever=retriever, agent_controller=controller, llm_enabled=True)
        qa_service = ChatService(controller=controller)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "stage": "smoke_init",
                    "message": f"Failed to initialize smoke dependencies: {exc}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1) from exc
    qa_session_store = InMemoryQASessionStore()
    cases = load_dataset(Path(args.dataset))
    results = []

    try:
        for case in cases:
            session_id = f"smoke_{case['case_id']}"
            trace_ids = []
            passed = True
            error = None

            def qa_handler(message: str) -> str | None:
                nonlocal trace_ids, session_id
                session = qa_session_store.get_or_create(session_id)
                result = qa_service.run_agent(
                    message,
                    history=session.history,
                    session_id=session.session_id,
                    session=session,
                    enable_session_memory=True,
                )
                commit_payload = qa_service.build_session_commit_payload(result, user_message=message)
                qa_session_store.commit_turn(
                    session,
                    user_message=commit_payload["user_message"],
                    assistant_message=commit_payload["assistant_message"],
                    trace_id=commit_payload["trace_id"],
                    entity_cache_updates=[
                        CachedAlignedEntity(**item) if isinstance(item, dict) else item
                        for item in commit_payload["entity_cache_updates"]
                    ],
                    recent_failure=FailureMemory(**commit_payload["recent_failure"])
                    if isinstance(commit_payload["recent_failure"], dict)
                    else commit_payload["recent_failure"],
                )
                trace_ids.append(result.trace.request_id)
                return result.answer

            try:
                last_response = None
                for turn in case["turns"]:
                    last_response = dialogue_service.chat(turn, session_id=session_id, qa_handler=qa_handler)
                    session_id = last_response["session_id"]
                if last_response is None:
                    raise RuntimeError("Smoke case has no turns.")
                if case["expected_action"] != last_response["action"]:
                    passed = False
                    error = f"Expected action {case['expected_action']}, got {last_response['action']}."
                elif case.get("min_recommendations", 0) > len(last_response["recommendations"]):
                    passed = False
                    error = f"Expected at least {case['min_recommendations']} recommendations."
                trace_payloads = [trace.model_dump() for trace in runtime.list_traces(session_id)] if session_id else []
                trace_ids.extend([item["request_id"] for item in trace_payloads if item["request_id"] not in trace_ids])
            except Exception as exc:
                passed = False
                error = str(exc)
                trace_payloads = [trace.model_dump() for trace in runtime.list_traces(session_id)] if session_id else []
                trace_ids.extend([item["request_id"] for item in trace_payloads if item["request_id"] not in trace_ids])

            results.append(
                {
                    "case_id": case["case_id"],
                    "session_id": session_id,
                    "passed": passed,
                    "error": error,
                    "trace_ids": trace_ids,
                    "traces": trace_payloads,
                }
            )
    finally:
        dialogue_service.close()
        qa_service.close()
        runtime.close()
        retriever.close()

    output_path = ROOT_DIR / "logs" / "eval" / f"dialogue_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_summary(output_path, {"cases": results})
    print(json.dumps({"output": str(output_path), "total_cases": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
