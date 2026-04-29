import json
import uuid

import streamlit as st

from .config import DEFAULT_CHAT_TITLE, DEFAULT_FOLDER_NAME, DEFAULT_FOLDERS, SAVE_FILE


def build_chat_record(title: str = DEFAULT_CHAT_TITLE, folder: str = DEFAULT_FOLDER_NAME) -> dict:
    return {
        "title": title or DEFAULT_CHAT_TITLE,
        "folder": folder,
        "messages": [],
    }


def create_chat(title: str = DEFAULT_CHAT_TITLE, folder: str = DEFAULT_FOLDER_NAME) -> str:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = build_chat_record(title=title, folder=folder)
    return chat_id


def ensure_active_chat() -> None:
    if not st.session_state.chats:
        st.session_state.current_chat_id = create_chat()
        return

    if st.session_state.current_chat_id not in st.session_state.chats:
        st.session_state.current_chat_id = next(iter(st.session_state.chats))


def save_data() -> None:
    payload = {
        "folders": st.session_state.folders,
        "chats": st.session_state.chats,
        "current_chat_id": st.session_state.current_chat_id,
    }

    with SAVE_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)


def load_data() -> None:
    if not SAVE_FILE.exists():
        return

    try:
        with SAVE_FILE.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return

    st.session_state.folders = payload.get("folders", list(DEFAULT_FOLDERS))
    st.session_state.chats = payload.get("chats", {})
    st.session_state.current_chat_id = payload.get("current_chat_id", "")
