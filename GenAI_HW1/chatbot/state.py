import streamlit as st

from .config import DEFAULT_FOLDERS
from .storage import ensure_active_chat, load_data


def initialize_session_state() -> None:
    if "folders" not in st.session_state:
        st.session_state.folders = list(DEFAULT_FOLDERS)
        st.session_state.chats = {}
        st.session_state.current_chat_id = ""
        load_data()

    ensure_active_chat()
