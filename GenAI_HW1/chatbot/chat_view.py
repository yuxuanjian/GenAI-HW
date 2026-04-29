import streamlit as st

from .llm import get_openai_client, stream_chat_completion
from .models import SidebarSettings
from .prompts import build_messages, build_system_prompt
from .storage import save_data


def render_chat_window(settings: SidebarSettings) -> None:
    curr_id = st.session_state.current_chat_id
    if curr_id not in st.session_state.chats:
        st.info("尚無可用對話。")
        return

    curr_chat = st.session_state.chats[curr_id]
    st.subheader(f"對話視窗: {curr_chat['title']}")

    for message in curr_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("請輸入訊息...")
    if not prompt:
        return

    curr_chat["messages"].append({"role": "user", "content": prompt})
    save_data()

    with st.chat_message("user"):
        st.markdown(prompt)

    full_system_prompt = build_system_prompt(
        base_prompt=settings.system_prompt,
        document_context=settings.document_context,
    )
    messages = build_messages(curr_chat["messages"], full_system_prompt)

    with st.chat_message("assistant"):
        try:
            client = get_openai_client()
            response_placeholder = st.empty()
            full_response = ""

            for content in stream_chat_completion(
                client=client,
                model=settings.selected_model,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            ):
                full_response += content
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            curr_chat["messages"].append({"role": "assistant", "content": full_response})
            save_data()
        except Exception as error:
            st.error(f"連線錯誤: {error}")
