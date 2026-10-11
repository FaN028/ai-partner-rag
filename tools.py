# ============================================================
# tools.py —— Agent 可调用的工具：函数本体 + 注册表 + 声明表
# ============================================================
import re
from datetime import datetime


def search_knowledge(query: str) -> str:
    """在用户上传的私有知识库中检索与问题相关的片段，返回最相关的原文内容。"""
    # 延迟导入：rag 在导入时会立刻加载 embedding 模型，
    # 放在文件顶部会让整个 tools.py 无法在终端里快速测试。
    try:
        import streamlit as st
        from rag import build_context_block

        chunks = st.session_state.knowledge_chunks
        vectors = st.session_state.knowledge_vectors
    except Exception:
        return "知识库尚未加载，请先在界面上传资料。"

    if not chunks or not vectors:
        return "知识库中没有找到相关内容。"

    _, selected = build_context_block(query, chunks, vectors)
    if not selected:
        return "知识库中没有找到相关内容。"

    # 只返回原文，不夹带"请你如何作答"的指令——工具负责取数据，不负责下命令。
    return "\n\n".join(f"【片段 {i}】\n{c}" for i, c in enumerate(selected, start=1))


def calculate(expression: str) -> str:
    """计算一个数学表达式并返回结果。只支持加减乘除和括号，不支持其他任何操作。"""
    # 第一道防线：白名单。整串必须只由数字和运算符号组成，出现其他字符一律拒绝。
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "只支持数字和 + - * / ( ) 运算。"

    # 第二道防线：禁用全部内置函数；第三道防线：异常兜底。
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"计算失败：{exc}"


def get_current_time() -> str:
    """获取当前的日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# —— 注册表：工具名 → 真正的函数（"后厨操作台"，给自己用的）——
TOOLS = {
    "search_knowledge": search_knowledge,
    "calculate": calculate,
    "get_current_time": get_current_time,
}


# —— 声明表：给模型看的说明书（"菜单"，模型只能看到这张）——
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
