from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from chatbot.chat_view import render_chat_window
from chatbot.config import APP_TITLE, PAGE_LAYOUT
from chatbot.sidebar import inject_sidebar_styles, render_sidebar
from chatbot.state import initialize_session_state


load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(page_title=APP_TITLE, layout=PAGE_LAYOUT)
st.title(APP_TITLE)

inject_sidebar_styles()
initialize_session_state()

sidebar_settings = render_sidebar()
render_chat_window(sidebar_settings)
