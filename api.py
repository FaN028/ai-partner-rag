# ============================================================
# api.py —— FastAPI 服务：将 Agent 调度能力暴露为 HTTP 接口
# ============================================================
import os

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

import config
from agent import run_agent
from tools import add_knowledge, clear_knowledge, knowledge_info

app = FastAPI(title=config.APP_TITLE, version="1.0.0")

SYSTEM_PROMPT = (
    "你是一个乐于助人的助手，可调用工具完成知识库检索、数学计算与时间查询。"
    "需要精确计算或需要知道当前时间时，请调用工具，不要自行推断。"
)

# 模块级初始化客户端；缺少 API Key 时置为 None，由接口按需返回 503。
_api_key = os.getenv(config.DEEPSEEK_API_KEY_ENV)
_client = (
    OpenAI(api_key=_api_key, base_url=config.DEEPSEEK_BASE_URL)
    if _api_key
    else None
)


class ChatRequest(BaseModel):
    """一次对话请求。"""

    question: str
    history: list[dict] | None = None


class ToolCallRecord(BaseModel):
    """单次工具调用记录。"""

    round: int
    tool: str
    args: dict
    result: str


class ChatResponse(BaseModel):
    """一次对话响应。"""

    answer: str
    trace: list[ToolCallRecord]


class KnowledgeRequest(BaseModel):
    """一份待入库的文本资料。"""

    source: str
    content: str


class KnowledgeResponse(BaseModel):
    """知识库当前状态。"""

    sources: list[str]
    chunks: int


@app.get("/health")
def health() -> dict:
    """健康检查：确认服务存活及其绑定的模型。"""
    return {"status": "ok", "model": config.MODEL_NAME}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """执行一轮 Agent 调度，返回最终答复与工具调用轨迹。"""
    if _client is None:
        raise HTTPException(
            status_code=503,
            detail=f"未配置环境变量 {config.DEEPSEEK_API_KEY_ENV}",
        )

    try:
        answer, trace = run_agent(
            _client,
            req.question,
            SYSTEM_PROMPT,
            req.history or [],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent 调用失败：{exc}") from exc

    return ChatResponse(
        answer=answer,
        trace=[ToolCallRecord(**step) for step in trace],
    )


@app.post("/knowledge", response_model=KnowledgeResponse)
def upload_knowledge(req: KnowledgeRequest) -> KnowledgeResponse:
    """把一份文本切块、向量化后加入服务端知识库。"""
    # 延迟导入：rag 在导入时会加载 embedding 与 rerank 模型。
    from rag import chunk_text, model

    chunks = chunk_text(req.content)
    if not chunks:
        raise HTTPException(status_code=400, detail="内容为空，未能切出任何片段。")

    add_knowledge(chunks, model.encode(chunks), source=req.source)
    return KnowledgeResponse(**knowledge_info())


@app.get("/knowledge", response_model=KnowledgeResponse)
def knowledge_status() -> KnowledgeResponse:
    """查看服务端知识库当前状态。"""
    return KnowledgeResponse(**knowledge_info())


@app.delete("/knowledge", response_model=KnowledgeResponse)
def reset_knowledge() -> KnowledgeResponse:
    """清空服务端知识库。"""
    clear_knowledge()
    return KnowledgeResponse(**knowledge_info())
