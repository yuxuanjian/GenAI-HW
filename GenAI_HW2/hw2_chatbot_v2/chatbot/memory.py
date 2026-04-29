from datetime import datetime

from .config import RECENT_HISTORY_MESSAGES


REMEMBER_PREFIXES = ("記住", "請記住", "幫我記住", "幫我記下", "請幫我記住")
FORGET_PREFIXES = ("忘記", "請忘記", "幫我忘記", "刪除記憶", "移除記憶")
NEGATED_MEMORY_MARKERS = ("不要記住", "不用記住", "別記住", "不要幫我記住")


def normalize_memory_text(text: str) -> str:
    return " ".join(text.strip(" ：:，,。.\n\t").split())


def extract_prefixed_content(prompt: str, prefixes: tuple[str, ...]) -> str:
    stripped_prompt = prompt.strip()
    for prefix in sorted(prefixes, key=len, reverse=True):
        if stripped_prompt.startswith(prefix):
            return normalize_memory_text(stripped_prompt[len(prefix):])
    return ""


def is_memory_directive(prompt: str) -> bool:
    if any(marker in prompt for marker in NEGATED_MEMORY_MARKERS):
        return False
    return bool(extract_prefixed_content(prompt, REMEMBER_PREFIXES) or extract_prefixed_content(prompt, FORGET_PREFIXES))


def append_pinned_memory(chat_record: dict, content: str) -> bool:
    content = normalize_memory_text(content)
    if not content:
        return False

    existing_memory = chat_record.get("pinned_memory", "")
    normalized_existing = existing_memory.replace(" ", "")
    if content.replace(" ", "") in normalized_existing:
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d")
    new_line = f"- {content}（{timestamp}）"
    chat_record["pinned_memory"] = f"{existing_memory.rstrip()}\n{new_line}".strip()
    return True


def remove_pinned_memory(chat_record: dict, content: str) -> bool:
    content = normalize_memory_text(content)
    if not content:
        return False

    lines = [line for line in chat_record.get("pinned_memory", "").splitlines() if line.strip()]
    if not lines:
        return False

    normalized_content = content.replace(" ", "")
    kept_lines = [line for line in lines if normalized_content not in line.replace(" ", "")]
    if len(kept_lines) == len(lines):
        return False

    chat_record["pinned_memory"] = "\n".join(kept_lines)
    return True


def apply_memory_directive(chat_record: dict, prompt: str) -> bool:
    if any(marker in prompt for marker in NEGATED_MEMORY_MARKERS):
        return False

    remember_content = extract_prefixed_content(prompt, REMEMBER_PREFIXES)
    if remember_content:
        return append_pinned_memory(chat_record, remember_content)

    forget_content = extract_prefixed_content(prompt, FORGET_PREFIXES)
    if forget_content:
        return remove_pinned_memory(chat_record, forget_content)

    return False


def _format_tool_outputs(message: dict) -> str:
    meta = message.get("meta", {})
    tool_outputs = meta.get("tool_outputs", [])
    if not tool_outputs:
        return ""

    tool_names = [tool_output.get("name", "unknown") for tool_output in tool_outputs]
    return f" [工具: {', '.join(tool_names)}]"


def _format_attachments(message: dict) -> str:
    attachments = message.get("attachments", [])
    image_names = [attachment.get("name", "image") for attachment in attachments if attachment.get("type") == "image"]
    if not image_names:
        return ""
    return f" [附加圖片: {', '.join(image_names)}]"


def format_messages_for_summary(messages: list[dict]) -> str:
    lines: list[str] = []
    for message in messages:
        content = (message.get("content") or "").strip()
        if not content:
            continue
        role = "使用者" if message.get("role") == "user" else "助理"
        lines.append(f"{role}{_format_attachments(message)}{_format_tool_outputs(message)}: {content}")
    return "\n".join(lines)


def fallback_summary(previous_summary: str, new_messages: list[dict]) -> str:
    recent_user_points = [
        message.get("content", "").strip()
        for message in new_messages
        if message.get("role") == "user" and message.get("content", "").strip()
    ]
    recent_assistant_points = [
        message.get("content", "").strip()
        for message in new_messages
        if message.get("role") == "assistant" and message.get("content", "").strip()
    ]

    lines: list[str] = []
    if previous_summary:
        lines.append(f"既有摘要：{previous_summary}")
    if recent_user_points:
        lines.append("近期使用者重點：" + "；".join(recent_user_points[-3:]))
    if recent_assistant_points:
        lines.append("近期助理回覆：" + "；".join(recent_assistant_points[-2:]))
    return "\n".join(lines).strip()


def maybe_refresh_memory(chat_record: dict, client, summary_model: str) -> bool:
    messages = chat_record.get("messages", [])
    checkpoint_target = max(len(messages) - RECENT_HISTORY_MESSAGES, 0)
    previous_checkpoint = chat_record.get("memory_checkpoint", 0)
    if checkpoint_target <= previous_checkpoint:
        return False

    previous_summary = chat_record.get("memory_summary", "")
    new_messages = messages[previous_checkpoint:checkpoint_target]
    transcript = format_messages_for_summary(new_messages)
    if not transcript:
        return False

    summary_prompt = [
        {
            "role": "system",
            "content": (
                "你是對話長期記憶壓縮器。"
                "請用繁體中文維護一份可供後續對話使用的持久記憶。"
                "優先保留：使用者長期目標、限制條件、偏好、重要名詞、已確認事實、已完成事項、待辦事項。"
                "不要保留寒暄或一次性廢話。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"舊摘要：\n{previous_summary or '（無）'}\n\n"
                f"新增對話：\n{transcript}\n\n"
                "請整合成新的長期記憶摘要，使用下列固定格式：\n"
                "1. 使用者目標\n"
                "2. 重要限制與偏好\n"
                "3. 關鍵事實與名詞\n"
                "4. 已完成進度\n"
                "5. 待辦事項\n"
                "若某欄沒有內容可寫「無」。總長控制在 12 行內。"
            ),
        },
    ]

    try:
        from .llm import complete_text

        updated_summary = complete_text(
            client=client,
            model=summary_model,
            messages=summary_prompt,
            temperature=0.1,
            max_tokens=500,
        ).strip()
    except Exception:
        updated_summary = fallback_summary(previous_summary, new_messages)

    chat_record["memory_summary"] = updated_summary
    chat_record["memory_checkpoint"] = checkpoint_target
    return True
