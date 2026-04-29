import streamlit as st

from .config import (
    DEFAULT_CHAT_TITLE,
    DEFAULT_FOLDER_NAME,
    DEFAULT_SYSTEM_PROMPT,
    MODEL_OPTIONS,
    UPLOADER_FILE_TYPES,
)
from .documents import build_document_context
from .models import DocumentContext, SidebarSettings
from .storage import create_chat, ensure_active_chat, save_data


SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.4rem !important;
}
[data-testid="stSidebar"] button {
    padding-top: 0px !important;
    padding-bottom: 0px !important;
    min-height: 2rem !important;
    margin-top: -2px !important;
}
[data-testid="stSidebar"] .stWidgetLabel {
    margin-bottom: -10px !important;
}
</style>
"""


def inject_sidebar_styles() -> None:
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)


def _create_chat_and_rerun(chat_title: str) -> None:
    new_id = create_chat(title=chat_title or DEFAULT_CHAT_TITLE, folder=DEFAULT_FOLDER_NAME)
    st.session_state.current_chat_id = new_id
    save_data()
    st.rerun()


def _delete_chat_and_rerun(chat_id: str) -> None:
    del st.session_state.chats[chat_id]
    ensure_active_chat()
    save_data()
    st.rerun()


def _delete_folder_and_chats(folder_name: str) -> None:
    st.session_state.chats = {
        chat_id: chat_data
        for chat_id, chat_data in st.session_state.chats.items()
        if chat_data["folder"] != folder_name
    }
    st.session_state.folders = [folder for folder in st.session_state.folders if folder != folder_name]
    ensure_active_chat()
    save_data()
    st.rerun()


def _move_folder_chats_to_default(folder_name: str) -> None:
    for chat_data in st.session_state.chats.values():
        if chat_data["folder"] == folder_name:
            chat_data["folder"] = DEFAULT_FOLDER_NAME

    st.session_state.folders = [folder for folder in st.session_state.folders if folder != folder_name]
    save_data()
    st.rerun()


def _render_management_panel() -> None:
    st.header("對話管理")

    with st.expander("管理功能", expanded=False):
        new_chat_name = st.text_input("新對話名稱", value=DEFAULT_CHAT_TITLE)
        if st.button("建立新對話", use_container_width=True):
            _create_chat_and_rerun(new_chat_name)

        st.markdown("---")

        with st.form("new_folder_form", clear_on_submit=True):
            new_folder_name = st.text_input("建立新群組")
            create_folder = st.form_submit_button("新增群組", use_container_width=True)

        if create_folder and new_folder_name and new_folder_name not in st.session_state.folders:
            st.session_state.folders.append(new_folder_name)
            save_data()
            st.rerun()


def _render_chat_groups() -> None:
    st.markdown("---")
    st.subheader("所有對話")

    for folder_name in st.session_state.folders:
        with st.expander(f"群組: {folder_name}", expanded=True):
            folder_chats = {
                chat_id: chat_data
                for chat_id, chat_data in st.session_state.chats.items()
                if chat_data["folder"] == folder_name
            }

            if not folder_chats:
                st.caption("尚無對話")

            for chat_id, chat_data in folder_chats.items():
                is_active = chat_id == st.session_state.current_chat_id
                button_label = f"[目前] {chat_data['title']}" if is_active else chat_data["title"]

                if st.button(button_label, key=f"select_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()

                if st.button(
                    f"刪除 {chat_data['title']}",
                    key=f"del_{chat_id}",
                    use_container_width=True,
                    type="secondary",
                ):
                    _delete_chat_and_rerun(chat_id)

            if folder_name != DEFAULT_FOLDER_NAME:
                with st.popover(f"刪除 {folder_name} 群組", use_container_width=True):
                    st.warning(f"要如何處理「{folder_name}」內的對話？")

                    if st.button(
                        "全部刪除 (包含對話)",
                        key=f"del_all_{folder_name}",
                        use_container_width=True,
                        type="primary",
                    ):
                        _delete_folder_and_chats(folder_name)

                    if st.button(
                        "僅刪除群組 (對話移至預設)",
                        key=f"keep_chats_{folder_name}",
                        use_container_width=True,
                    ):
                        _move_folder_chats_to_default(folder_name)


def _render_current_chat_settings() -> None:
    st.markdown("---")
    st.header("設定面板")

    curr_id = st.session_state.current_chat_id
    if curr_id not in st.session_state.chats:
        return

    current_chat = st.session_state.chats[curr_id]
    original_title = current_chat["title"]
    updated_title = st.text_input(
        "重新命名目前對話",
        value=original_title,
        key=f"title_input_{curr_id}",
    )
    if updated_title != original_title:
        current_chat["title"] = updated_title
        save_data()

    current_folder = current_chat["folder"]
    folder_index = st.session_state.folders.index(current_folder) if current_folder in st.session_state.folders else 0
    new_folder_selection = st.selectbox("將目前對話移至群組", options=st.session_state.folders, index=folder_index)
    if new_folder_selection != current_folder:
        current_chat["folder"] = new_folder_selection
        save_data()
        st.rerun()


def _render_document_controls(selected_model: str) -> DocumentContext:
    uploaded_file = st.file_uploader("上傳參考檔案", type=UPLOADER_FILE_TYPES)
    if not uploaded_file:
        return DocumentContext()

    try:
        document_context = build_document_context(uploaded_file, selected_model)
        if document_context.total_chunks > 1:
            st.info(f"檔案已針對 {selected_model} 切分為 {document_context.total_chunks} 個語意銜接片段。")
            chunk_index = st.selectbox(
                "選擇目前閱讀段落",
                range(document_context.total_chunks),
                format_func=lambda idx: f"第 {idx + 1} 部分",
            )
            document_context.select_chunk(chunk_index)

        st.success(f"檔案 {uploaded_file.name} 讀取成功")
        return document_context
    except Exception as error:
        st.error(f"檔案解析失敗: {error}")
        return DocumentContext()


def render_sidebar() -> SidebarSettings:
    with st.sidebar:
        _render_management_panel()
        _render_chat_groups()
        _render_current_chat_settings()

        st.markdown("---")
        selected_model = st.selectbox("選擇模型", MODEL_OPTIONS, index=0)
        system_prompt = st.text_area("System Prompt", value=DEFAULT_SYSTEM_PROMPT)
        document_context = _render_document_controls(selected_model)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.number_input("Max Tokens", 100, 100000, 25000)

    return SidebarSettings(
        selected_model=selected_model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        document_context=document_context,
    )
