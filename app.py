# ============================================================
# app.py —— 程序入口：配置页面 + AI 调用 + 初始化状态 + 主程序
# ============================================================
import streamlit as st
import config

# 必须最先调用（在导入 rag 之前，因为 rag 会在导入时就加载模型）
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

import os
from typing import Iterator

from openai import OpenAI

from session import generate_session_name, save_session
from rag import build_context_block, model
from agent import run_agent, save_trace
import ui


def build_system_prompt(nick_name: str, nature: str) -> str:
    """根据用户设置生成系统提示词。"""
    return f"""
你叫 {nick_name}，现在是用户的知识库助手，请完全代入知识库助手角色。
规则：
    1. 每次只回1条消息
    2. 禁止任何场景或状态描述性文字
    3. 匹配用户的语言
    4. 回复简短，像微信聊天一样
    5. 有需要的话可以用emoji表情
    6. 用符合知识库助手设定的方式对话
    7. 回复的内容要充分体现知识库助手的性格特征
知识库助手性格：
    - {nature}
你必须严格遵守上述规则来回复用户。
""".strip()


def create_client() -> OpenAI | None:
    """创建 DeepSeek API 客户端。"""
    api_key = os.getenv(config.DEEPSEEK_API_KEY_ENV)

    if not api_key:
        st.error(
            f"没有找到环境变量 {config.DEEPSEEK_API_KEY_ENV}。"
            "请先配置 DeepSeek API Key。"
        )
        return None

    return OpenAI(
        api_key=api_key,
        base_url=config.DEEPSEEK_BASE_URL,
    )


def stream_chat_response(client: OpenAI, messages: list[dict]) -> Iterator[str]:
    """调用模型，并逐段返回 AI 回复。"""
    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=messages,
        stream=True,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    for chunk in response:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content
        if content:
            yield content


def run_agent_turn(client: OpenAI, prompt: str, system_prompt: str, history: list[dict]) -> None:
    """Agent 模式下执行一轮对话：调用 agent.run_agent 并渲染结果。

    与「知识库模式」的差异：
      1. 不预先检索知识库，是否检索由模型自行决定；
      2. 无流式输出，需等待全部工具调用结束后一次性返回。
    """
    try:
        with st.spinner("模型正在思考、按需调用工具…"):
            answer, trace = run_agent(client, prompt, system_prompt, history)
    except Exception as exc:
        st.error(f"Agent 调用失败：{exc}")
        return

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_session()

    # 轨迹落盘，供事后复盘与问题定位。
    save_trace(prompt, trace)

    # 工具调用轨迹面板（与「参考来源」面板同构）。
    if trace:
        with st.expander(f"🔧 工具调用轨迹（{len(trace)} 次）"):
            for step in trace:
                st.markdown(f"**第 {step['round']} 轮 · `{step['tool']}`**")
                st.caption(f"参数：{step['args']}")
                st.code(str(step["result"])[:800], language=None)


def init_session_state() -> None:
    """初始化程序运行期间需要保存的数据。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "nick_name" not in st.session_state:
        st.session_state.nick_name = config.DEFAULT_NICK_NAME

    if "nature" not in st.session_state:
        st.session_state.nature = config.DEFAULT_NATURE

    if "current_session" not in st.session_state:
        st.session_state.current_session = generate_session_name()

    # —— 知识库相关的状态 ——
    if "knowledge_chunks" not in st.session_state:
        st.session_state.knowledge_chunks = []
    if "knowledge_vectors" not in st.session_state:
        st.session_state.knowledge_vectors = []
    if "knowledge_sources" not in st.session_state:
        st.session_state.knowledge_sources = []
    if "knowledge_ids" not in st.session_state:
        st.session_state.knowledge_ids = set()
    if "knowledge_clear_counter" not in st.session_state:
        st.session_state.knowledge_clear_counter = 0

    # —— 记忆（多轮对话记忆）相关状态 ——
    if "memory_chunks" not in st.session_state:
        st.session_state.memory_chunks = []
    if "memory_vectors" not in st.session_state:
        st.session_state.memory_vectors = []

    # —— 对话模式：知识库模式（默认，和原先一样）/ Agent 模式（模型自己决定调不调工具）——
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = config.MODE_KB


def main() -> None:
    """程序入口。"""
    st.title(config.APP_TITLE)

    init_session_state()

    if config.LOGO_PATH.exists():
        st.logo(str(config.LOGO_PATH))

    ui.render_chat_history()
    ui.render_sidebar()

    client = create_client()
    if client is None:
        return

    prompt = st.chat_input("请输入您要问的问题")
    if not prompt:
        return

    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    # 把用户这句话也存进"长期记忆"
    st.session_state.memory_chunks.append(prompt)
    st.session_state.memory_vectors.append(model.encode(prompt))

    system_prompt = build_system_prompt(
        st.session_state.nick_name,
        st.session_state.nature,
    )

    # Agent 模式：本轮交由 agent.run_agent 处理，处理完直接返回。
    # history 排除本轮提问——它作为 user_query 单独传入，否则会重复。
    if st.session_state.chat_mode == config.MODE_AGENT:
        run_agent_turn(client, prompt, system_prompt, st.session_state.messages[:-1])
        return

    # 知识库模式：先检索知识库，再流式生成回答。
    context, cited = build_context_block(
        prompt,
        st.session_state.knowledge_chunks,
        st.session_state.knowledge_vectors,
    )
    # 检索历史记忆
    memory_context, _ = build_context_block(
        prompt,
        st.session_state.memory_chunks,
        st.session_state.memory_vectors,
    )

    if context:
        system_prompt = system_prompt + "\n\n" + context
    if memory_context:
        system_prompt = system_prompt + "\n\n【你的记忆】之前聊过的相关内容：\n" + memory_context

    request_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *st.session_state.messages,
    ]

    response_placeholder = st.empty()
    full_response = ""

    try:
        for content in stream_chat_response(client, request_messages):
            full_response += content
            with response_placeholder.container():
                with st.chat_message("assistant"):
                    st.write(full_response)
    except Exception as exc:
        st.error(f"调用 AI 失败：{exc}")
        return

    if full_response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
        })
        save_session()

        # 显示"引用来源"
        if cited:
            with st.expander("📚 参考来源"):
                for i, chunk in enumerate(cited, start=1):
                    st.write(f"【片段 {i}】{chunk}")
    else:
        st.warning("AI 没有返回可显示的内容。")


if __name__ == "__main__":
    main()
