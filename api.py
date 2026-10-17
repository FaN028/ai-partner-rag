# ============================================================
# api.py —— FastAPI 服务：将 Agent 调度能力暴露为 HTTP 接口
# ============================================================
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

import config
import knowledge
from agent import run_agent
from tools import add_knowledge, clear_knowledge, knowledge_info

app = FastAPI(title=config.APP_TITLE, version="1.0.0")

# 允许浏览器跨域调用。
# 学生/演示项目先全放开；若正式对外，应把 allow_origins 收成具体域名白名单。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def bind_client(request: Request, call_next):
    """把请求头里的客户端标识写入上下文。

    工具函数 search_knowledge 拿不到请求信息，只能通过上下文读取，
    这样不同客户端各自拥有独立的知识库。
    """
    knowledge.set_current_client(request.headers.get("X-Client-Id"))
    return await call_next(request)

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
    """健康检查：确认服务存活、绑定的模型与向量后端。

    容器编排可用它判断实例是否就绪；同时把后端类型暴露出来，
    避免「以为在用 Chroma、实际退回了 NumPy」这类静默降级。
    """
    store = knowledge.get_store()
    return {
        "status": "ok",
        "model": config.MODEL_NAME,
        "vector_backend": store.backend,
        "chunks": store.count(),
    }


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
