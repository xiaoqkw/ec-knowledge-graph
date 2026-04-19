from pydantic import BaseModel


class Question(BaseModel):
    """前端提交的问题请求体。"""
    message: str

class Answer(BaseModel):
    """问答接口返回的响应体。"""
    message: str
