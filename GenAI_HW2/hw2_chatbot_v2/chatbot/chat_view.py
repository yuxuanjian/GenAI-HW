import hashlib
import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from .config import (
    ATTACHMENT_DIR,
    DEFAULT_MODEL_CONTEXT_WINDOW,
    IMAGE_FILE_TYPES,
    MODEL_CONTEXT_WINDOWS,
    VISION_IMAGE_JPEG_QUALITY,
    VISION_IMAGE_MAX_DIMENSION,
)
from .history import build_precise_history_answer
from .llm import get_openai_client, stream_chat_completion
from .mcp_client import call_tools_via_mcp
from .memory import FORGET_PREFIXES, extract_prefixed_content, maybe_refresh_memory, remove_pinned_memory
from .memory_store import (
    apply_structured_memory_directive,
    maybe_update_structured_memory,
    retrieve_structured_memory_context,
)
from .models import SidebarSettings, ToolOutput
from .prompts import build_api_messages, build_system_prompt
from .routing import decide_route
from .storage import save_data
from .tool_query import (
    build_failed_query_planning_output,
    extract_document_context,
    plan_augmented_search_query,
)
from .tools.calculator import build_calculator_tool_output
from .tools.history_lookup import build_history_lookup_tool_output
from .tools.retrieval import build_document_retrieval_tool_output
from .tools.web_search import build_web_search_tool_output


ATTACHMENT_TOOLBAR_CSS = """
<style>
[data-testid="stAppViewBlockContainer"] {
    padding-bottom: 8.5rem !important;
}

.st-key-attachment_toolbar {
    position: fixed;
    bottom: 6.75rem;
    left: 25.5rem;
    z-index: 1000;
    width: fit-content;
    max-width: 100%;
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 0;
    margin-top: 0.5rem;
}

.st-key-attachment_toolbar [data-testid="stPopover"] button {
    min-height: 2.1rem;
    border-radius: 999px;
    padding: 0 0.8rem;
}

.st-key-web_search_toggle button {
    min-height: 1.65rem !important;
    height: 1.65rem !important;
    width: 5.25rem !important;
    padding: 0 0.6rem !important;
    font-size: 0.88rem !important;
    border-radius: 999px !important;
    white-space: nowrap !important;
    line-height: 1 !important;
}

.st-key-attachment_toolbar [data-testid="stHorizontalBlock"] {
    align-items: end;
}

@media (max-width: 900px) {
    .st-key-attachment_toolbar {
        left: 1rem;
        bottom: 4.5rem;
    }
}
</style>
"""


def optimize_image_for_vision(file_bytes: bytes) -> tuple[bytes, str]:
    image = Image.open(io.BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)

    if image.mode not in ("RGB", "L"):
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[-1] if "A" in image.getbands() else None)
        image = rgb_image
    elif image.mode == "L":
        image = image.convert("RGB")

    image.thumbnail((VISION_IMAGE_MAX_DIMENSION, VISION_IMAGE_MAX_DIMENSION))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=VISION_IMAGE_JPEG_QUALITY, optimize=True)
    return buffer.getvalue(), ".jpg"


def build_persistent_image_attachment(chat_id: str, uploaded_image) -> dict:
    original_bytes = uploaded_image.getvalue()
    optimized_bytes, optimized_suffix = optimize_image_for_vision(original_bytes)
    digest = hashlib.sha256(optimized_bytes).hexdigest()
    chat_attachment_dir = ATTACHMENT_DIR / chat_id
    chat_attachment_dir.mkdir(parents=True, exist_ok=True)
    saved_path = chat_attachment_dir / f"{digest}{optimized_suffix}"
    if not saved_path.exists():
        saved_path.write_bytes(optimized_bytes)

    return {
        "type": "image",
        "name": uploaded_image.name,
        "path": str(saved_path),
        "size_kb": round(len(optimized_bytes) / 1024, 1),
    }


def chat_has_image_context(chat_record: dict) -> bool:
    for message in reversed(chat_record.get("messages", [])):
        for attachment in message.get("attachments", []):
            if attachment.get("type") == "image":
                return True
    return False


def get_latest_image_attachment(chat_record: dict) -> dict | None:
    for message in reversed(chat_record.get("messages", [])):
        for attachment in reversed(message.get("attachments", [])):
            if attachment.get("type") == "image":
                return attachment
    return None


def toggle_web_search(chat_id: str, chat_record: dict) -> None:
    chat_record["web_search_enabled"] = not chat_record.get("web_search_enabled", False)
    save_data()
    st.rerun()


def render_attachment_uploader(chat_id: str, chat_record: dict) -> None:
    pending_image = chat_record.get("pending_image_attachment")
    latest_image = get_latest_image_attachment(chat_record)
    web_search_enabled = chat_record.get("web_search_enabled", False)

    with st.container(key="attachment_toolbar"):
        image_column, search_column = st.columns([1, 0.72], gap="small")

        with image_column:
            with st.popover("Image", use_container_width=False):
                uploaded_image = st.file_uploader(
                    "Upload image to this chat",
                    type=IMAGE_FILE_TYPES,
                    key=f"main_image_upload_{chat_id}_{chat_record.get('image_upload_nonce', 0)}",
                )

                if uploaded_image:
                    chat_record["pending_image_attachment"] = build_persistent_image_attachment(chat_id, uploaded_image)

                pending_image = chat_record.get("pending_image_attachment")
                if pending_image:
                    st.image(pending_image["path"], caption=pending_image["name"], use_container_width=True)
                    size_kb = pending_image.get("size_kb")
                    size_text = f" ({size_kb} KB)" if size_kb else ""
                    st.success(f"Ready to send: {pending_image['name']}{size_text}")
                    if st.button("Remove pending image", key=f"clear_pending_image_{chat_id}", use_container_width=True):
                        chat_record["pending_image_attachment"] = None
                        chat_record["image_upload_nonce"] = chat_record.get("image_upload_nonce", 0) + 1
                        st.rerun()
                elif latest_image:
                    st.caption(f"Current image context: {latest_image['name']}")
                    with st.expander("Preview current image", expanded=False):
                        st.image(latest_image["path"], caption=latest_image["name"], use_container_width=True)
                else:
                    st.caption("No image attached in this chat.")

        with search_column:
            button_type = "primary" if web_search_enabled else "secondary"
            with st.container(key="web_search_toggle"):
                if st.button(
                    "Search",
                    key=f"toggle_web_search_{chat_id}",
                    help="Turn web search on or off for this chat",
                    use_container_width=False,
                    type=button_type,
                ):
                    toggle_web_search(chat_id, chat_record)



def render_message(message: dict) -> None:
    with st.chat_message(message.get("role", "assistant")):
        for attachment in message.get("attachments", []):
            if attachment.get("type") == "image":
                image_path = attachment.get("path", "")
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption=attachment.get("name", "image"), use_container_width=True)

        st.markdown(message.get("content", ""))

        meta = message.get("meta", {})
        if meta:
            model = meta.get("model")
            tools = meta.get("tools", [])
            route_reason = meta.get("reason", "")
            tool_outputs = meta.get("tool_outputs", [])
            tool_text = ", ".join(tools) if tools else "無"
            caption = f"Model: {model or 'unknown'} | Tools: {tool_text}"
            tool_metadata_caption = format_tool_metadata_caption(tool_outputs)
            if tool_metadata_caption:
                caption += f" | {tool_metadata_caption}"
            if route_reason:
                caption += f" | Route: {route_reason}"
            st.caption(caption)

            if tool_outputs:
                with st.expander("Tool Results", expanded=False):
                    for tool_output in tool_outputs:
                        st.markdown(f"**{tool_output['name']}**")
                        transport = tool_output.get("metadata", {}).get("transport")
                        if transport:
                            st.caption(f"Transport: {transport}")
                        st.code(tool_output["content"], language="text")


def format_tool_metadata_caption(tool_outputs: list[dict | ToolOutput]) -> str:
    providers = []
    transports = []
    for tool_output in tool_outputs:
        if isinstance(tool_output, ToolOutput):
            metadata = tool_output.metadata
        else:
            metadata = tool_output.get("metadata", {})

        provider = metadata.get("provider")
        transport = metadata.get("transport")
        if provider:
            providers.append(provider)
        if transport:
            transports.append(transport)

    parts = []
    if providers:
        parts.append(f"Provider: {', '.join(dict.fromkeys(providers))}")
    if transports:
        parts.append(f"Transport: {', '.join(dict.fromkeys(transports))}")
    return " | ".join(parts)


def get_model_context_window(model_name: str) -> int:
    return MODEL_CONTEXT_WINDOWS.get(model_name, DEFAULT_MODEL_CONTEXT_WINDOW)


def emit_thinking_progress(progress_callback, message: str) -> None:
    if progress_callback:
        progress_callback(message)


def execute_tools(prompt: str, tool_names: list[str]) -> list[ToolOutput]:
    curr_id = st.session_state.current_chat_id
    current_chat = st.session_state.chats.get(curr_id, {})
    document_library = current_chat.get("document_library", {})

    if not tool_names:
        return []

    try:
        return call_tools_via_mcp(prompt, tool_names, curr_id)
    except Exception:
        pass

    outputs: list[ToolOutput] = []
    for tool_name in tool_names:
        if tool_name == "calculator":
            output = build_calculator_tool_output(prompt)
        elif tool_name == "document_retrieval":
            output = build_document_retrieval_tool_output(prompt, document_library=document_library)
        elif tool_name == "web_search":
            output = build_web_search_tool_output(prompt)
        elif tool_name == "history_lookup":
            output = build_history_lookup_tool_output(prompt, current_chat.get("messages", [])[:-1])
        else:
            continue

        output.metadata = {**output.metadata, "transport": "local-fallback"}
        outputs.append(output)

    return outputs


def execute_tools_with_planning(
    prompt: str,
    tool_names: list[str],
    current_chat: dict,
    settings: SidebarSettings,
    use_vision: bool,
    client=None,
    progress_callback=None,
) -> tuple[list[ToolOutput], object | None]:
    if not tool_names:
        return [], client

    outputs: list[ToolOutput] = []
    remaining_tools = list(dict.fromkeys(tool_names))

    pre_web_tools = [tool_name for tool_name in ("calculator", "document_retrieval") if tool_name in remaining_tools]
    if pre_web_tools:
        emit_thinking_progress(progress_callback, f"正在呼叫工具：{', '.join(pre_web_tools)}。")
        outputs.extend(execute_tools(prompt, pre_web_tools))
        for tool_name in pre_web_tools:
            remaining_tools.remove(tool_name)

    if "web_search" in remaining_tools:
        web_query = prompt
        planning_output = None
        candidate_queries = [prompt]
        document_output = next((output for output in outputs if output.name == "document_retrieval"), None)
        document_context = extract_document_context(document_output)
        image_attachment = get_latest_image_attachment(current_chat) if use_vision else None

        if image_attachment or document_context:
            try:
                emit_thinking_progress(progress_callback, "正在根據圖片或文件線索規劃搜尋查詢。")
                client = client or get_openai_client()
                planner_model = settings.vision_model if image_attachment else settings.large_model
                web_query, planning_output = plan_augmented_search_query(
                    client=client,
                    model=planner_model,
                    user_query=prompt,
                    image_attachment=image_attachment,
                    document_context=document_context,
                )
                candidate_queries = planning_output.metadata.get("candidate_queries", [web_query]) if planning_output else [web_query]
            except Exception as error:
                planning_output = build_failed_query_planning_output(prompt, error)
                web_query = prompt
                candidate_queries = [prompt]

        if planning_output:
            outputs.append(planning_output)

        unique_queries = []
        for query in candidate_queries:
            normalized_query = " ".join(str(query).split())
            if normalized_query and normalized_query not in unique_queries:
                unique_queries.append(normalized_query)

        max_query_count = 2 if (image_attachment or document_context) else 1
        selected_queries = unique_queries[:max_query_count] or [web_query]
        for query_index, search_query in enumerate(selected_queries, start=1):
            emit_thinking_progress(progress_callback, f"正在執行 web search ({query_index}/{len(selected_queries)})：{search_query}")
            web_outputs = execute_tools(search_query, ["web_search"])
            for output in web_outputs:
                output.metadata = {
                    **output.metadata,
                    "original_query": prompt,
                    "augmented_query": web_query,
                    "query_augmented": web_query != prompt,
                    "candidate_query": search_query,
                    "candidate_query_index": query_index,
                }
            outputs.extend(web_outputs)
        remaining_tools.remove("web_search")

    for tool_name in ("history_lookup",):
        if tool_name not in remaining_tools:
            continue
        emit_thinking_progress(progress_callback, "正在查詢對話歷史。")
        outputs.extend(execute_tools(prompt, [tool_name]))
        remaining_tools.remove(tool_name)

    for tool_name in remaining_tools:
        emit_thinking_progress(progress_callback, f"正在呼叫工具：{tool_name}。")
        outputs.extend(execute_tools(prompt, [tool_name]))

    return outputs, client


def render_chat_window(settings: SidebarSettings) -> None:
    st.markdown(ATTACHMENT_TOOLBAR_CSS, unsafe_allow_html=True)

    curr_id = st.session_state.current_chat_id
    if curr_id not in st.session_state.chats:
        st.info("尚無可用對話。")
        return

    current_chat = st.session_state.chats[curr_id]
    st.subheader(f"對話視窗: {current_chat['title']}")

    for message in current_chat["messages"]:
        render_message(message)

    render_attachment_uploader(curr_id, current_chat)
    prompt = st.chat_input("請輸入訊息...")
    if not prompt:
        return

    attachments = [current_chat["pending_image_attachment"]] if current_chat.get("pending_image_attachment") else []
    user_message = {
        "role": "user",
        "content": prompt,
        "attachments": attachments,
    }
    current_chat["messages"].append(user_message)
    apply_structured_memory_directive(prompt, curr_id)
    forget_content = extract_prefixed_content(prompt, FORGET_PREFIXES)
    if forget_content:
        remove_pinned_memory(current_chat, forget_content)
    current_chat["pending_image_attachment"] = None
    current_chat["image_upload_nonce"] = current_chat.get("image_upload_nonce", 0) + 1
    save_data()

    render_message(user_message)

    document_names = [document.get("name", "") for document in current_chat.get("document_library", {}).values()]
    route = decide_route(
        prompt=prompt,
        has_new_image=bool(attachments),
        has_image_context=chat_has_image_context(current_chat),
        web_search_enabled=current_chat.get("web_search_enabled", False),
        document_names=document_names,
        small_model=settings.small_model,
        large_model=settings.large_model,
        vision_model=settings.vision_model,
    )
    history_messages = current_chat.get("messages", [])[:-1]

    with st.chat_message("assistant"):
        thinking_status = None
        try:
            thinking_status = st.status("思考中...", expanded=True)
            thinking_status.write(
                f"已完成路由：模型 `{route.selected_model}`；工具 `{', '.join(route.tools) if route.tools else '無'}`。"
            )

            def show_thinking_progress(message: str) -> None:
                if thinking_status:
                    thinking_status.write(message)

            if not route.tools:
                thinking_status.write("本輪不需要工具，準備直接生成回答。")

            client = None
            tool_outputs, client = execute_tools_with_planning(
                prompt=prompt,
                tool_names=route.tools,
                current_chat=current_chat,
                settings=settings,
                use_vision=route.use_vision,
                client=client,
                progress_callback=show_thinking_progress,
            )
            direct_history_answer = None
            if route.tools == ["history_lookup"]:
                thinking_status.write("正在根據精準對話順序產生答案。")
                direct_history_answer = build_precise_history_answer(prompt, history_messages)

            if direct_history_answer:
                thinking_status.update(label="已完成對話歷史查詢", state="complete", expanded=False)
                st.markdown(direct_history_answer)
                st.caption(
                    f"Model: deterministic-history | Tools: {', '.join(route.tools)} | Route: {route.reason} 精準順序題直接依據 history lookup 作答。"
                )
                assistant_message = {
                    "role": "assistant",
                    "content": direct_history_answer,
                    "meta": {
                        "model": "deterministic-history",
                        "tools": route.tools,
                        "reason": f"{route.reason} 精準順序題直接依據 history lookup 作答。",
                        "tool_outputs": [
                            {"name": output.name, "content": output.content, "metadata": output.metadata}
                            for output in tool_outputs
                        ],
                    },
                }
                current_chat["messages"].append(assistant_message)
                save_data()
                return

            thinking_status.write("正在整理長期記憶與系統上下文。")
            client = client or get_openai_client()
            maybe_refresh_memory(current_chat, client=client, summary_model=settings.large_model)
            structured_memory_context = retrieve_structured_memory_context(prompt)
            system_prompt = build_system_prompt(
                base_prompt=settings.system_prompt,
                memory_summary=current_chat.get("memory_summary", ""),
                route=route,
                pinned_memory=current_chat.get("pinned_memory", ""),
                structured_memory_context=structured_memory_context,
            )
            api_messages = build_api_messages(
                chat_messages=current_chat["messages"],
                system_prompt=system_prompt,
                tool_outputs=tool_outputs,
                include_images=route.use_vision,
            )

            thinking_status.write("正在生成最終回答。")
            response_placeholder = st.empty()
            full_response = ""
            effective_max_tokens = min(settings.max_tokens, get_model_context_window(route.selected_model))
            for content in stream_chat_completion(
                client=client,
                model=route.selected_model,
                messages=api_messages,
                temperature=settings.temperature,
                max_tokens=effective_max_tokens,
            ):
                full_response += content
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            thinking_status.update(label="已完成", state="complete", expanded=False)
            caption = f"Model: {route.selected_model} | Tools: {', '.join(route.tools) if route.tools else '無'}"
            tool_metadata_caption = format_tool_metadata_caption(tool_outputs)
            if tool_metadata_caption:
                caption += f" | {tool_metadata_caption}"
            if effective_max_tokens != settings.max_tokens:
                caption += f" | Max Tokens capped: {effective_max_tokens:,}"
            caption += f" | Route: {route.reason}"
            st.caption(caption)

            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "meta": {
                    "model": route.selected_model,
                    "tools": route.tools,
                    "reason": route.reason,
                    "tool_outputs": [
                        {"name": output.name, "content": output.content, "metadata": output.metadata}
                        for output in tool_outputs
                    ],
                },
            }
            current_chat["messages"].append(assistant_message)
            maybe_update_structured_memory(
                client=client,
                model=settings.large_model,
                chat_id=curr_id,
                user_message=prompt,
                assistant_message=full_response,
                tool_outputs=tool_outputs,
            )
            save_data()
        except Exception as error:
            if thinking_status:
                thinking_status.update(label="發生錯誤", state="error", expanded=True)
            st.error(f"連線錯誤: {error}")
