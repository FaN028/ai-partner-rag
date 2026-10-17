# ============================================================
# tools.py —— Agent 工具集：函数实现 + 注册表 + 声明表
# ============================================================
import re
from datetime import datetime

import knowledge

# ============================================================
# 知识库来源：界面模式读 Streamlit 会话，HTTP 模式读服务端存储
# （服务端存储按客户端隔离，并落盘到 config.KNOWLEDGE_DIR）
# ============================================================
def add_knowledge(chunks: list, vectors, source: str = "") -> int:
    """向服务端知识库追加片段与向量并立即落盘，返回追加后的片段总数。"""
    store = knowledge.get_store()
    total = store.add(chunks, vectors, source)
    store.save()
    return total


def clear_knowledge() -> None:
    """清空服务端知识库（含已落盘文件）。"""
    knowledge.get_store().clear()


def knowledge_info() -> dict:
    """返回服务端知识库的当前状态。"""
    return knowledge.get_store().info()


def _resolve_knowledge():
    """取当前可用的知识库视图：(chunks, vectors, searcher)。

    - 界面模式：数据在 Streamlit 会话状态里，直接返回 (chunks, vectors, None)；
    - 服务端模式：数据在向量库后端里，只交出 searcher，避免把全量片段与
      向量搬进内存——这正是换成 Chroma 后端后省掉的那部分开销。
    """
    try:
        import streamlit as st

        chunks = st.session_state.knowledge_chunks
        vectors = st.session_state.knowledge_vectors
        if len(chunks) > 0 and len(vectors) > 0:
            return chunks, vectors, None
    except Exception:
        pass

    store = knowledge.get_store()
    if not store.ready:
        return None, None, None
    return None, None, store.search


def search_knowledge(query: str) -> str:
    """在私有知识库中检索与 query 相关的原文片段。

    返回拼接后的原文片段；知识库为空或未命中时返回提示文本。
    """
    chunks, vectors, searcher = _resolve_knowledge()
    # 先判断知识库是否为空：为空时直接返回，避免白白加载数百 MB 的检索模型。
    if searcher is None and (not chunks or not vectors):
        return "知识库中还没有内容。"

    # 延迟导入：rag 模块在导入时即加载 embedding 与 rerank 模型，
    # 置于模块顶层会使本文件无法脱离模型独立导入与测试。
    try:
        from rag import build_context_block
    except Exception:
        return "检索组件尚未就绪。"

    _, selected = build_context_block(query, chunks, vectors, searcher=searcher)
    if not selected:
        return "知识库中没有找到相关内容。"

    # 仅回填原文，不附加任何作答指令，避免工具越权干预模型生成策略。
    return "\n\n".join(f"【片段 {i}】\n{c}" for i, c in enumerate(selected, start=1))


def calculate(expression: str) -> str:
    """计算四则运算表达式并返回字符串结果。

    表达式由模型生成，直接求值存在任意代码执行风险，
    因此依次执行白名单校验、剥离内置函数、异常兜底三层防护。
    """
    # 白名单：整串仅允许数字与 + - * / ( ) 及空白字符。
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "只支持数字和 + - * / ( ) 运算。"

    try:
        # __builtins__ 置空，阻断 import / open / exec 等内置能力。
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"计算失败：{exc}"


def get_current_time() -> str:
    """获取当前日期与时间，格式 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 工具注册表：工具名 -> 可调用对象，供调度端按名派发。
TOOLS = {
    "search_knowledge": search_knowledge,
    "calculate": calculate,
    "get_current_time": get_current_time,
}


# 工具声明表：下发至模型的 Function Calling Schema，模型仅可见此表。
# 约束：name 须与 TOOLS 的键一致，parameters 中的参数名须与函数签名一致。
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "在用户上传的私有知识库中检索相关原文片段。"
                "当用户的问题可能涉及用户上传过的资料、文档、笔记时使用。"
                "返回的是原文片段，不是答案。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要在知识库中检索的问题或关键词。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "计算一个数学表达式并返回数值结果。只支持加减乘除和括号。"
                "当用户需要精确算数（四则运算、百分比等）时使用，不要自己心算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 (3+5)*2。",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "获取当前的日期和时间。"
                "当用户问到今天几号、现在几点、当前日期时间时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]
