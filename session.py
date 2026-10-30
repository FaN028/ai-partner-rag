# ============================================================
# session.py —— 会话管理：生成、保存、加载、删除会话
# ============================================================
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

import config


def generate_session_name() -> str:
    """生成一个基于当前时间的唯一会话名称。"""
    return datetime.now(config.TIMEZONE).strftime("%Y-%m-%d_%H-%M-%S")


def get_session_path(session_name: str) -> Path:
    """根据会话名称生成对应的 JSON 文件路径。"""
    return config.SESSIONS_DIR / f"{session_name}.json"


def save_session() -> bool:
    """保存当前会话到 JSON 文件。"""
    session_name = st.session_state.current_session
    if not session_name:
        return False

    session_data = {
        "nick_name": st.session_state.nick_name,
        "nature": st.session_state.nature,
        "messages": st.session_state.messages,
        "current_session": session_name,
    }

    try:
        config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        session_path = get_session_path(session_name)
        with session_path.open("w", encoding="utf-8") as file:
            json.dump(session_data, file, ensure_ascii=False, indent=2)
        return True
    except OSError as exc:
        st.error(f"保存会话失败：{exc}")
        return False


def load_sessions() -> list[str]:
    """读取所有历史会话名称，并按名称倒序排列。"""
    if not config.SESSIONS_DIR.exists():
        return []

    session_list = [
        path.stem
        for path in config.SESSIONS_DIR.iterdir()
        if path.is_file() and path.suffix == ".json"
    ]
    session_list.sort(reverse=True)
    return session_list


def load_session(session_name: str) -> bool:
    """读取指定会话并恢复到 session_state。"""
    session_path = get_session_path(session_name)

    if not session_path.exists():
        st.warning("找不到这个会话文件。")
        return False

    try:
        with session_path.open("r", encoding="utf-8") as file:
            session_data = json.load(file)

        st.session_state.nick_name = session_data.get("nick_name", config.DEFAULT_NICK_NAME)
        st.session_state.nature = session_data.get("nature", config.DEFAULT_NATURE)
        st.session_state.messages = session_data.get("messages", [])
        st.session_state.current_session = session_name
        return True
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"加载会话失败：{exc}")
        return False


def delete_session(session_name: str) -> bool:
    """删除指定会话。"""
    session_path = get_session_path(session_name)

    try:
        if session_path.exists():
            session_path.unlink()

        if session_name == st.session_state.current_session:
            st.session_state.messages = []
            st.session_state.current_session = generate_session_name()
        return True
    except OSError as exc:
        st.error(f"删除会话失败：{exc}")
        return False
