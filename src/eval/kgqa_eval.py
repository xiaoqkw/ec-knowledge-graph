import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import LOG_DIR
from web.memory import CachedAlignedEntity, FailureMemory, InMemoryQASessionStore
from web.service import ChatService


TEMPLATE_CYPHERS = {
    "brand_products": """
        MATCH (tm:Trademark {name: $param_0})<-[:Belong]-(spu:SPU)
        RETURN DISTINCT spu.name AS name
        ORDER BY name
        LIMIT 20
    """,
    "category_brands": """
        MATCH (spu:SPU)-[:Belong]->(:Category3 {name: $param_0})
        MATCH (spu)-[:Belong]->(tm:Trademark)
        RETURN DISTINCT tm.name AS name
        ORDER BY name
        LIMIT 20
    """,
    "product_skus": """
        MATCH (sku:SKU)-[:Belong]->(:SPU {name: $param_0})
        RETURN DISTINCT sku.name AS name
        ORDER BY name
        LIMIT 20
    """,
}


def load_dataset(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def normalize_text(text: str) -> str:
    return str(text or "").lower()


def answer_keyword_hit(answer: str, keywords: list[str]) -> bool:
    normalized_answer = normalize_text(answer)
    return any(normalize_text(keyword) in normalized_answer for keyword in keywords)


def has_answer_keywords(keywords: list[str]) -> bool:
    return any(str(keyword).strip() for keyword in keywords)


def coverage_flags(required_entities: list[str], executed_params: dict) -> tuple[bool, bool]:
    if not required_entities:
        return True, True
    values = set(executed_params.values())
    any_hit = any(entity in values for entity in required_entities)
    all_hit = all(entity in values for entity in required_entities)
    return any_hit, all_hit


def run_template_baseline(service: ChatService, sample: dict) -> dict:
    cypher_query = TEMPLATE_CYPHERS[sample["gold_cypher_type"]]
    executed_params = dict(sample["template_params"])
    unsafe_cypher = service._is_unsafe_cypher(cypher_query)
    execution_success = False
    execution_error = None
    query_result = []
    non_empty_result = False
    try:
        query_result = service._execute_cypher(cypher_query, executed_params)
        execution_success = True
        non_empty_result = bool(query_result)
    except Exception as exc:
        execution_error = str(exc)

    answer = json.dumps(query_result, ensure_ascii=False) if query_result else "当前图谱中没有找到相关信息。"
    any_hit, all_hit = coverage_flags(sample.get("required_entities", []), executed_params)
    return {
        "question": sample["question"],
        "baseline": "template",
        "gold_cypher_type": sample["gold_cypher_type"],
        "must_execute": bool(sample.get("must_execute", True)),
        "has_answer_keywords": has_answer_keywords(sample.get("answer_keywords", [])),
        "raw_cypher_output": "",
        "repaired_cypher_output": "",
        "cypher_query": cypher_query,
        "entities_to_align": [],
        "aligned_entities": [],
        "executed_params": executed_params,
        "parse_success_raw": True,
        "parse_success_repaired": True,
        "cypher_query_present": True,
        "unsafe_cypher": unsafe_cypher,
        "execution_success": execution_success,
        "execution_error": execution_error,
        "non_empty_result": non_empty_result,
        "query_result": query_result,
        "answer": answer,
        "answer_keyword_hit": answer_keyword_hit(answer, sample.get("answer_keywords", [])),
        "entity_any_coverage_hit": any_hit,
        "entity_all_coverage_hit": all_hit,
    }


def _commit_eval_session_turn(store: InMemoryQASessionStore, session, service: ChatService, result, *, user_message: str) -> None:
    commit_payload = service.build_session_commit_payload(result, user_message=user_message)
    store.commit_turn(
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


def _trace_result_to_record(question: str, result) -> dict:
    trace = result.trace
    metadata = trace.metadata
    execution_error = None
    for record in trace.tool_calls:
        if not record.ok and record.metadata.get("error"):
            execution_error = record.metadata["error"]
            break
    return {
        "question": question,
        "raw_cypher_output": metadata.get("planner_raw_output", ""),
        "repaired_cypher_output": metadata.get("planner_repaired_output", ""),
        "parse_success_raw": bool(metadata.get("planner_parse_success_raw", False)),
        "parse_success_repaired": bool(metadata.get("planner_parse_success_repaired", False)),
        "cypher_query_present": bool(metadata.get("cypher_query")),
        "parsed_payload": metadata.get("parsed_payload"),
        "cypher_query": metadata.get("cypher_query", ""),
        "entities_to_align": metadata.get("entities_to_align", []),
        "aligned_entities": metadata.get("aligned_entities", []),
        "executed_params": metadata.get("executed_params", {}),
        "unsafe_cypher": "unsafe_query_blocked" in trace.failure_tags,
        "execution_success": any(call.tool_name == "graph_query_tool" and call.ok for call in trace.tool_calls),
        "execution_error": execution_error,
        "query_result": metadata.get("query_result", []),
        "non_empty_result": bool(metadata.get("query_result", [])),
        "answer": result.answer,
        "trace_id": trace.request_id,
    }


def run_llm_baseline(
    service: ChatService,
    sample: dict,
    baseline: str,
    *,
    enable_session_memory: bool,
    session_store: InMemoryQASessionStore | None = None,
    session=None,
) -> dict:
    if enable_session_memory:
        if session_store is None or session is None:
            raise ValueError("Session memory baseline requires both session_store and session.")
        result = service.run_agent(
            sample["question"],
            history=session.history,
            session_id=session.session_id,
            align_entities=(baseline == "full"),
            session=session,
            enable_session_memory=True,
        )
        _commit_eval_session_turn(session_store, session, service, result, user_message=sample["question"])
        trace = _trace_result_to_record(sample["question"], result)
    else:
        trace = service.trace_chat(
            sample["question"],
            align_entities=(baseline == "full"),
            enable_session_memory=False,
        )
    any_hit, all_hit = coverage_flags(sample.get("required_entities", []), trace["executed_params"])
    return {
        "question": sample["question"],
        "baseline": baseline,
        "gold_cypher_type": sample["gold_cypher_type"],
        "must_execute": bool(sample.get("must_execute", True)),
        "has_answer_keywords": has_answer_keywords(sample.get("answer_keywords", [])),
        "raw_cypher_output": trace["raw_cypher_output"],
        "repaired_cypher_output": trace["repaired_cypher_output"],
        "cypher_query": trace["cypher_query"],
        "entities_to_align": trace["entities_to_align"],
        "aligned_entities": trace["aligned_entities"],
        "executed_params": trace["executed_params"],
        "parse_success_raw": trace["parse_success_raw"],
        "parse_success_repaired": trace["parse_success_repaired"],
        "cypher_query_present": trace["cypher_query_present"],
        "unsafe_cypher": trace["unsafe_cypher"],
        "execution_success": trace["execution_success"],
        "execution_error": trace["execution_error"],
        "non_empty_result": trace["non_empty_result"],
        "query_result": trace["query_result"],
        "answer": trace["answer"],
        "answer_keyword_hit": answer_keyword_hit(trace["answer"], sample.get("answer_keywords", [])),
        "entity_any_coverage_hit": any_hit,
        "entity_all_coverage_hit": all_hit,
    }


def compute_metrics(records: list[dict]) -> dict:
    total = len(records)
    must_execute_records = [record for record in records if record.get("must_execute", True)]
    keyword_records = [record for record in records if record.get("has_answer_keywords", False)]

    def rate(key: str, subset: list[dict] | None = None) -> float:
        target = subset if subset is not None else records
        return sum(1 for record in target if record[key]) / len(target) if target else 0.0

    return {
        "total": total,
        "must_execute_total": len(must_execute_records),
        "answer_keyword_total": len(keyword_records),
        "raw_json_parse_success_rate": rate("parse_success_raw"),
        "repaired_json_parse_success_rate": rate("parse_success_repaired"),
        "cypher_query_present_rate": rate("cypher_query_present"),
        "cypher_execution_success_rate": rate("execution_success", must_execute_records),
        "non_empty_result_rate": rate("non_empty_result", must_execute_records),
        "answer_keyword_hit_rate": rate("answer_keyword_hit", keyword_records),
        "unsafe_cypher_rate": rate("unsafe_cypher"),
        "entity_any_coverage_rate": rate("entity_any_coverage_hit"),
        "entity_all_coverage_rate": rate("entity_all_coverage_hit"),
    }


def write_logs(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


def print_report(results: dict) -> None:
    print("| baseline | total | must_execute | keyword_total | raw_parse | repaired_parse | query_present | execute | non_empty | keyword_hit | unsafe | entity_any | entity_all |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for baseline, metrics in results.items():
        print(
            "| "
            f"{baseline} | {metrics['total']} | {metrics['must_execute_total']} | {metrics['answer_keyword_total']} | {metrics['raw_json_parse_success_rate']:.4f} | "
            f"{metrics['repaired_json_parse_success_rate']:.4f} | {metrics['cypher_query_present_rate']:.4f} | "
            f"{metrics['cypher_execution_success_rate']:.4f} | {metrics['non_empty_result_rate']:.4f} | "
            f"{metrics['answer_keyword_hit_rate']:.4f} | {metrics['unsafe_cypher_rate']:.4f} | "
            f"{metrics['entity_any_coverage_rate']:.4f} | {metrics['entity_all_coverage_rate']:.4f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KGQA baselines.")
    parser.add_argument(
        "--dataset",
        default=str(ROOT_DIR / "data" / "eval" / "kgqa.jsonl"),
    )
    parser.add_argument(
        "--baseline",
        choices=["template", "full", "ablation", "all"],
        default="all",
    )
    parser.add_argument("--enable-session-memory", action="store_true")
    args = parser.parse_args()

    baselines = ["template", "full", "ablation"] if args.baseline == "all" else [args.baseline]
    dataset = load_dataset(Path(args.dataset))
    needs_llm = any(baseline in {"full", "ablation"} for baseline in baselines)
    service = ChatService(llm_enabled=needs_llm)
    try:
        results = {}
        for baseline in baselines:
            session_store = InMemoryQASessionStore() if args.enable_session_memory else None
            session = (
                session_store.get_or_create(f"kgqa_eval_session_{baseline}")
                if session_store is not None
                else None
            )
            records = []
            for sample in dataset:
                if baseline == "template":
                    record = run_template_baseline(service, sample)
                else:
                    record = run_llm_baseline(
                        service,
                        sample,
                        baseline,
                        enable_session_memory=args.enable_session_memory,
                        session_store=session_store,
                        session=session,
                    )
                records.append(record)
            memory_suffix = "memory_on" if args.enable_session_memory else "memory_off"
            write_logs(LOG_DIR / "eval" / f"kgqa_{baseline}_{memory_suffix}.jsonl", records)
            results[baseline] = compute_metrics(records)
        memory_suffix = "memory_on" if args.enable_session_memory else "memory_off"
        write_summary(LOG_DIR / "eval" / f"kgqa_summary_{memory_suffix}.json", results)
        print_report(results)
    finally:
        service.close()


if __name__ == "__main__":
    main()
