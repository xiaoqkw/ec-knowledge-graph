import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import AgentController, AgentRuntime, AgentResources
from configuration.config import WEB_STATIC_DIR
from dialogue import DialogueService
from dialogue.retrieval import PhoneGuideRetriever
from web.memory import CachedAlignedEntity, FailureMemory, InMemoryQASessionStore
from web.schemas import (
    AgentResponse,
    AgentTurnRequest,
    Answer,
    DialogueTurnRequest,
    DialogueTurnResponse,
    PlanSummaryItem,
    Question,
    ReplayRequest,
)
from web.service import ChatService


app = FastAPI(title="电商知识图谱问答与导购服务")
app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")

shared_retriever: PhoneGuideRetriever | None = None
agent_runtime: AgentRuntime | None = None
agent_controller: AgentController | None = None
dialogue_service: DialogueService | None = None
qa_service: ChatService | None = None
qa_service_error: str | None = None
dialogue_service_error: str | None = None
qa_session_store = InMemoryQASessionStore()


def get_agent_controller() -> AgentController:
    global shared_retriever, agent_runtime, agent_controller
    if agent_controller is not None:
        return agent_controller
    if shared_retriever is None:
        shared_retriever = PhoneGuideRetriever()
    if agent_runtime is None:
        agent_runtime = AgentRuntime(resources=AgentResources())
    agent_controller = AgentController(agent_runtime, retriever=shared_retriever, enable_explain=True)
    return agent_controller


def get_dialogue_service() -> DialogueService:
    global shared_retriever, dialogue_service, dialogue_service_error
    if dialogue_service is not None:
        return dialogue_service
    if dialogue_service_error is not None:
        raise RuntimeError(dialogue_service_error)

    try:
        controller = get_agent_controller()
        if shared_retriever is None:
            shared_retriever = PhoneGuideRetriever()
        dialogue_service = DialogueService(retriever=shared_retriever, agent_controller=controller)
        return dialogue_service
    except Exception as exc:
        dialogue_service_error = str(exc)
        raise RuntimeError(dialogue_service_error) from exc


def get_qa_service() -> ChatService | None:
    global qa_service, qa_service_error
    if qa_service is not None:
        return qa_service
    if qa_service_error is not None:
        return None

    try:
        qa_service = ChatService(controller=get_agent_controller())
    except Exception as exc:
        qa_service_error = str(exc)
        return None
    return qa_service


def run_qa(message: str, history: list[dict[str, str]] | None = None) -> str | None:
    service = get_qa_service()
    if service is None:
        return None
    return service.chat(message, history=history)


@app.get("/")
def read_root():
    return RedirectResponse("/static/index.html")


@app.post("/api/chat")
def chat_api(question: Question) -> Answer:
    service = get_qa_service()
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=qa_service_error or "Knowledge graph QA is not enabled in the current environment.",
        )
    session = qa_session_store.get_or_create(question.session_id)
    result = service.run_agent(
        question.message,
        history=session.history,
        session_id=session.session_id,
        session=session,
        enable_session_memory=True,
    )
    commit_payload = service.build_session_commit_payload(result, user_message=question.message)
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
    return Answer(message=result.answer, session_id=session.session_id)


@app.post("/api/agent/chat")
def agent_chat_api(turn: AgentTurnRequest, debug: bool = False) -> AgentResponse:
    service = get_qa_service()
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=qa_service_error or "Knowledge graph QA is not enabled in the current environment.",
        )
    session = qa_session_store.get_or_create(turn.session_id)
    result = service.run_agent(
        turn.message,
        history=session.history,
        session_id=session.session_id,
        session=session,
        enable_session_memory=True,
    )
    commit_payload = service.build_session_commit_payload(result, user_message=turn.message)
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
    response = AgentResponse(
        answer=result.answer,
        session_id=result.session_id,
        trace_id=result.trace.request_id,
        plan_summary=[PlanSummaryItem(tool=item.tool_name) for item in result.trace.plan],
        latency_ms=result.trace.total_latency_ms,
        fallback_used=result.trace.fallback_used,
    )
    if debug:
        response.trace = result.trace.dict()
    return response


@app.get("/api/agent/traces/{trace_id}")
def get_agent_trace(trace_id: str) -> dict:
    if agent_runtime is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    trace = agent_runtime.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace.dict()


@app.get("/api/agent/traces")
def list_agent_traces(session_id: str) -> list[dict]:
    if agent_runtime is None:
        return []
    return [trace.dict() for trace in agent_runtime.list_traces(session_id)]


@app.post("/api/agent/replay")
def replay_agent_traces(payload: ReplayRequest) -> list[dict]:
    if agent_runtime is None:
        return []
    return [trace.dict() for trace in agent_runtime.replay_traces(payload.trace_ids)]


@app.post("/api/dialogue/chat")
def dialogue_chat_api(turn: DialogueTurnRequest) -> DialogueTurnResponse:
    service = get_dialogue_service()
    result = service.chat(
        turn.message,
        session_id=turn.session_id,
        qa_handler=run_qa,
    )
    return DialogueTurnResponse(**result)


@app.on_event("shutdown")
def close_services() -> None:
    global shared_retriever, agent_runtime, agent_controller, dialogue_service, qa_service
    global qa_service_error, dialogue_service_error
    if dialogue_service is not None:
        dialogue_service.close()
    if qa_service is not None:
        qa_service.close()
    if agent_runtime is not None:
        agent_runtime.close()
    if shared_retriever is not None:
        shared_retriever.close()
    dialogue_service = None
    qa_service = None
    agent_controller = None
    agent_runtime = None
    shared_retriever = None
    qa_service_error = None
    dialogue_service_error = None


if __name__ == "__main__":
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000)
