import json
import uuid

import streamlit as st

from .config import ATTACHMENT_DIR, DATA_DIR, DEFAULT_CHAT_TITLE, DEFAULT_FOLDER_NAME, DEFAULT_FOLDERS, SAVE_FILE, UPLOAD_DIR


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)


def build_chat_record(title: str = DEFAULT_CHAT_TITLE, folder: str = DEFAULT_FOLDER_NAME) -> dict:
    return {
        "title": title or DEFAULT_CHAT_TITLE,
        "folder": folder,
        "messages": [],
        "pinned_memory": "",
        "memory_summary": "",
        "memory_checkpoint": 0,
        "document_library": {},
        "pending_image_attachment": None,
        "web_search_enabled": False,
        "document_upload_nonce": 0,
        "image_upload_nonce": 0,
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


def normalize_loaded_chat(chat_record: dict) -> dict:
    normalized = build_chat_record(
        title=chat_record.get("title", DEFAULT_CHAT_TITLE),
        folder=chat_record.get("folder", DEFAULT_FOLDER_NAME),
    )
    normalized["messages"] = chat_record.get("messages", [])
    normalized["pinned_memory"] = chat_record.get("pinned_memory", "")
    normalized["memory_summary"] = chat_record.get("memory_summary", "")
    normalized["memory_checkpoint"] = chat_record.get("memory_checkpoint", 0)
    normalized["document_library"] = chat_record.get("document_library", {})
    normalized["pending_image_attachment"] = chat_record.get("pending_image_attachment")
    normalized["web_search_enabled"] = chat_record.get("web_search_enabled", False)
    normalized["document_upload_nonce"] = chat_record.get("document_upload_nonce", 0)
    normalized["image_upload_nonce"] = chat_record.get("image_upload_nonce", 0)
    return normalized


def save_data() -> None:
    ensure_data_dirs()

    payload = {
        "folders": st.session_state.folders,
        "chats": st.session_state.chats,
        "current_chat_id": st.session_state.current_chat_id,
    }

    with SAVE_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_data() -> None:
    ensure_data_dirs()
    if not SAVE_FILE.exists():
        return

    try:
        with SAVE_FILE.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return

    st.session_state.folders = payload.get("folders", list(DEFAULT_FOLDERS))
    st.session_state.chats = {
        chat_id: normalize_loaded_chat(chat_record)
        for chat_id, chat_record in payload.get("chats", {}).items()
    }
    st.session_state.current_chat_id = payload.get("current_chat_id", "")

    # Backward compatibility: migrate the old top-level document library into the
    # currently active chat so existing uploads are not lost after the refactor.
    legacy_document_library = payload.get("document_library", {})
    if legacy_document_library and st.session_state.chats:
        target_chat_id = st.session_state.current_chat_id
        if target_chat_id not in st.session_state.chats:
            target_chat_id = next(iter(st.session_state.chats))
        target_chat = st.session_state.chats[target_chat_id]
        if not target_chat.get("document_library"):
            target_chat["document_library"] = legacy_document_library
