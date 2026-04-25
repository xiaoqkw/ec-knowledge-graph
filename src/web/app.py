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

from configuration.config import WEB_STATIC_DIR
from dialogue import DialogueService
from web.memory import InMemoryQASessionStore
from web.schemas import Answer, DialogueTurnRequest, DialogueTurnResponse, Question
from web.service import ChatService


app = FastAPI(title="电商知识图谱问答与导购服务")
app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")

dialogue_service = DialogueService()
qa_service: ChatService | None = None
qa_service_error: str | None = None
qa_session_store = InMemoryQASessionStore()


def get_qa_service() -> ChatService | None:
    global qa_service, qa_service_error
    if qa_service is not None:
        return qa_service
    if qa_service_error is not None:
        return None

    try:
        qa_service = ChatService()
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
    result = service.chat(question.message, history=session.history)
    qa_session_store.save_turn(session, question.message, result)
    return Answer(message=result, session_id=session.session_id)


@app.post("/api/dialogue/chat")
def dialogue_chat_api(turn: DialogueTurnRequest) -> DialogueTurnResponse:
    result = dialogue_service.chat(
        turn.message,
        session_id=turn.session_id,
        qa_handler=run_qa,
    )
    return DialogueTurnResponse(**result)


@app.on_event("shutdown")
def close_services() -> None:
    dialogue_service.close()
    if qa_service is not None:
        qa_service.close()


if __name__ == "__main__":
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000)
