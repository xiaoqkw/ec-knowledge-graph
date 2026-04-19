import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles


CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import WEB_STATIC_DIR
from web.schemas import Answer, Question
from web.service import ChatService


app = FastAPI(title="电商知识图谱问答服务")
app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")

service = ChatService()


@app.get("/")
def read_root():
    """访问根路径时直接跳转到静态前端页面。"""
    return RedirectResponse("/static/index.html")


@app.post("/api/chat")
def chat_api(question: Question) -> Answer:
    """接收前端问题文本，返回知识图谱问答结果。"""
    result = service.chat(question.message)
    return Answer(message=result)


if __name__ == "__main__":
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000)
