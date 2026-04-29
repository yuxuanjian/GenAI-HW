import os
from importlib.util import find_spec
from pathlib import Path

import streamlit as st

from .config import (
    DEFAULT_CHAT_TITLE,
    DEFAULT_FOLDER_NAME,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_CONTEXT_WINDOW,
    DEFAULT_SYSTEM_PROMPT,
    DOCUMENT_FILE_TYPES,
    GENERAL_MODEL_OPTIONS,
    MODEL_CONTEXT_WINDOWS,
)
from .documents import delete_document, ingest_uploaded_documents
from .memory_store import (
    MEMORY_TYPE_LABELS,
    MEMORY_TYPES,
    delete_memory_item,
    load_structured_memory_store,
)
from .models import SidebarSettings
from .storage import create_chat, ensure_active_chat, save_data


SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.45rem !important;
}
[data-testid="stSidebar"] button {
    min-height: 2rem !important;
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
                label = f"[目前] {chat_data['title']}" if is_active else chat_data["title"]

                if st.button(label, key=f"select_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()

                if st.button(f"刪除 {chat_data['title']}", key=f"del_{chat_id}", use_container_width=True, type="secondary"):
                    _delete_chat_and_rerun(chat_id)

            if folder_name != DEFAULT_FOLDER_NAME:
                with st.popover(f"刪除 {folder_name} 群組", use_container_width=True):
                    st.warning(f"要如何處理「{folder_name}」內的對話？")
                    if st.button("全部刪除 (包含對話)", key=f"del_all_{folder_name}", use_container_width=True, type="primary"):
                        _delete_folder_and_chats(folder_name)
                    if st.button("僅刪除群組 (對話移至預設)", key=f"keep_chats_{folder_name}", use_container_width=True):
                        _move_folder_chats_to_default(folder_name)


def _render_current_chat_settings() -> None:
    st.markdown("---")
    st.header("目前對話設定")

    curr_id = st.session_state.current_chat_id
    if curr_id not in st.session_state.chats:
        return

    current_chat = st.session_state.chats[curr_id]
    original_title = current_chat["title"]
    updated_title = st.text_input("重新命名目前對話", value=original_title, key=f"title_input_{curr_id}")
    if updated_title != original_title:
        current_chat["title"] = updated_title
        save_data()

    current_folder = current_chat["folder"]
    folder_index = st.session_state.folders.index(current_folder) if current_folder in st.session_state.folders else 0
    new_folder = st.selectbox("移至群組", options=st.session_state.folders, index=folder_index)
    if new_folder != current_folder:
        current_chat["folder"] = new_folder
        save_data()
        st.rerun()


def _render_document_library() -> None:
    st.markdown("---")
    st.header("文件知識庫")
    curr_id = st.session_state.current_chat_id
    if curr_id not in st.session_state.chats:
        st.caption("目前沒有可用對話。")
        return

    current_chat = st.session_state.chats[curr_id]

    with st.form("document_upload_form", clear_on_submit=True):
        uploaded_docs = st.file_uploader(
            "匯入到目前對話的 PDF / DOCX / TXT / 程式碼檔",
            type=DOCUMENT_FILE_TYPES,
            accept_multiple_files=True,
            key=f"document_upload_{curr_id}_{current_chat.get('document_upload_nonce', 0)}",
        )
        upload_clicked = st.form_submit_button("匯入文件", use_container_width=True)

    if upload_clicked and uploaded_docs:
        result = ingest_uploaded_documents(curr_id, current_chat, uploaded_docs)
        if result["imported"]:
            st.success("已匯入: " + "、".join(result["imported"]))
        for message in result["skipped"]:
            st.info(message)
        for message in result["errors"]:
            st.error(message)
        current_chat["document_upload_nonce"] = current_chat.get("document_upload_nonce", 0) + 1
        save_data()

    documents = current_chat.get("document_library", {})
    if not documents:
        st.caption("目前對話還沒有文件。")
        return

    st.caption(f"目前對話已建立 {len(documents)} 份文件索引")
    for doc_id, document in documents.items():
        with st.expander(document["name"], expanded=False):
            st.write(f"類型: `{document['file_type']}`")
            st.write(f"Chunk 數量: `{document['chunk_count']}`")
            st.write(f"上傳時間: `{document['uploaded_at']}`")
            saved_path = Path(document.get("saved_path", ""))
            if saved_path.exists():
                st.caption(f"本地檔案: {saved_path.name}")
            if st.button("刪除文件", key=f"delete_doc_{doc_id}", use_container_width=True, type="secondary"):
                delete_document(current_chat, doc_id)
                st.rerun()


def _render_model_settings() -> tuple[str, str, str, str, float, int]:
    st.markdown("---")
    st.header("模型與參數")

    default_small = os.getenv("DEFAULT_SMALL_MODEL", GENERAL_MODEL_OPTIONS[0])
    default_large = os.getenv("DEFAULT_LARGE_MODEL", GENERAL_MODEL_OPTIONS[-1])
    default_vision = os.getenv("DEFAULT_VISION_MODEL", default_large)

    small_index = GENERAL_MODEL_OPTIONS.index(default_small) if default_small in GENERAL_MODEL_OPTIONS else 0
    large_index = GENERAL_MODEL_OPTIONS.index(default_large) if default_large in GENERAL_MODEL_OPTIONS else len(GENERAL_MODEL_OPTIONS) - 1

    small_model = st.selectbox("小模型", GENERAL_MODEL_OPTIONS, index=small_index)
    large_model = st.selectbox("大模型", GENERAL_MODEL_OPTIONS, index=large_index)
    vision_model = st.text_input("Vision Model", value=default_vision)
    st.caption("如果圖片辨識效果差，優先檢查這裡填的是不是真的支援 vision 的模型，而不是一般文字模型。")
    old_default_prompt = "你是一個具備工具調用能力的專業助理。請使用繁體中文回答。"
    if "system_prompt_input" not in st.session_state or st.session_state.system_prompt_input == old_default_prompt:
        st.session_state.system_prompt_input = DEFAULT_SYSTEM_PROMPT
    system_prompt = st.text_area("System Prompt", key="system_prompt_input")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.5, 0.1)
    selected_context_window = max(
        MODEL_CONTEXT_WINDOWS.get(model_name, DEFAULT_MODEL_CONTEXT_WINDOW)
        for model_name in {small_model, large_model, vision_model}
        if model_name
    )
    if "max_tokens_input" not in st.session_state or st.session_state.max_tokens_input > DEFAULT_MAX_OUTPUT_TOKENS:
        st.session_state.max_tokens_input = DEFAULT_MAX_OUTPUT_TOKENS
    max_tokens = st.number_input(
        "Max Output Tokens",
        min_value=200,
        max_value=DEFAULT_MAX_OUTPUT_TOKENS,
        step=1000,
        key="max_tokens_input",
    )
    st.caption(f"輸出上限固定為 32k；目前選到的模型最大 context window：{selected_context_window:,} tokens")

    return small_model, large_model, vision_model, system_prompt, temperature, max_tokens


def _format_latest_tool_status(current_chat: dict) -> tuple[str, str]:
    for message in reversed(current_chat.get("messages", [])):
        tool_outputs = message.get("meta", {}).get("tool_outputs", [])
        if not tool_outputs:
            continue

        transports = [
            output.get("metadata", {}).get("transport", "")
            for output in tool_outputs
            if output.get("metadata", {}).get("transport")
        ]
        providers = [
            output.get("metadata", {}).get("provider", "")
            for output in tool_outputs
            if output.get("metadata", {}).get("provider")
        ]
        transport_text = ", ".join(dict.fromkeys(transports)) if transports else "尚無紀錄"
        provider_text = ", ".join(dict.fromkeys(providers)) if providers else "無"
        return transport_text, provider_text

    return "尚無工具呼叫紀錄", "無"


def _render_mcp_status_panel() -> None:
    curr_id = st.session_state.current_chat_id
    current_chat = st.session_state.chats.get(curr_id, {})
    transport_text, provider_text = _format_latest_tool_status(current_chat)

    with st.expander("MCP 狀態", expanded=False):
        mcp_available = find_spec("mcp") is not None
        st.write(f"MCP 套件：`{'available' if mcp_available else 'missing'}`")
        st.write("工具呼叫策略：`MCP stdio first -> local fallback`")
        st.write("MCP tools：`calculator`, `web_search`, `document_retrieval`, `history_lookup`")
        st.write(f"最近一次 transport：`{transport_text}`")
        st.write(f"最近一次 provider：`{provider_text}`")

        if "local-fallback" in transport_text:
            st.warning("最近一次工具呼叫走 local fallback，代表 MCP server 啟動或 mcp 套件可能有問題。")
        elif "mcp-stdio" in transport_text:
            st.success("最近一次工具呼叫有走 MCP stdio。")
        elif not mcp_available:
            st.warning("目前環境找不到 mcp 套件，工具會退回 local fallback。")


def _render_memory_panel() -> None:
    curr_id = st.session_state.current_chat_id
    if curr_id not in st.session_state.chats:
        return

    current_chat = st.session_state.chats[curr_id]
    memory_summary = current_chat.get("memory_summary", "")
    pinned_memory = current_chat.get("pinned_memory", "")
    with st.expander("Long-Term Memory 摘要", expanded=False):
        updated_pinned_memory = st.text_area(
            "Current Chat Pinned Memory（僅目前 chat；明確「記住」會寫入下方跨 chat Structured Memory）",
            value=pinned_memory,
            key=f"pinned_memory_{curr_id}",
            height=120,
        )
        if updated_pinned_memory != pinned_memory:
            current_chat["pinned_memory"] = updated_pinned_memory
            save_data()

        st.markdown("---")
        if memory_summary:
            st.write(memory_summary)
        else:
            st.caption("目前尚未累積到需要摘要的對話。")

        st.markdown("---")
        st.subheader("Structured Memory Store")
        store = load_structured_memory_store()
        items = store.get("items", [])
        if not items:
            st.caption("目前尚無跨 chat 結構化記憶。")
            return

        counts = {
            memory_type: len([item for item in items if item.get("type") == memory_type])
            for memory_type in MEMORY_TYPES
        }
        count_text = " / ".join(
            f"{MEMORY_TYPE_LABELS[memory_type]} {counts[memory_type]}"
            for memory_type in MEMORY_TYPES
            if counts[memory_type]
        )
        st.caption(count_text)

        for memory_type in MEMORY_TYPES:
            typed_items = [item for item in items if item.get("type") == memory_type]
            if not typed_items:
                continue

            st.markdown(f"**{MEMORY_TYPE_LABELS[memory_type]} ({len(typed_items)})**")
            for item in sorted(typed_items, key=lambda entry: entry.get("updated_at", ""), reverse=True):
                st.markdown(f"- {item.get('content', '')}")
                st.caption(
                    f"id: `{item.get('id', '')}` | confidence: `{item.get('confidence', 0):.2f}` | updated: `{item.get('updated_at', '')}`"
                )
                if st.button(
                    "刪除這筆記憶",
                    key=f"delete_memory_{item.get('id', '')}",
                    use_container_width=True,
                    type="secondary",
                ):
                    delete_memory_item(item.get("id", ""))
                    st.rerun()


def render_sidebar() -> SidebarSettings:
    with st.sidebar:
        _render_management_panel()
        _render_chat_groups()
        _render_current_chat_settings()
        _render_document_library()
        small_model, large_model, vision_model, system_prompt, temperature, max_tokens = _render_model_settings()
        _render_mcp_status_panel()
        _render_memory_panel()

    current_chat = st.session_state.chats.get(st.session_state.current_chat_id, {})
    return SidebarSettings(
        small_model=small_model,
        large_model=large_model,
        vision_model=vision_model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        pending_image=current_chat.get("pending_image_attachment"),
    )
