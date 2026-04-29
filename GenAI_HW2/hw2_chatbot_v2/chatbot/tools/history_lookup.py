from ..history import build_episodic_memory_context
from ..models import ToolOutput


def build_history_lookup_tool_output(prompt: str, messages: list[dict]) -> ToolOutput:
    context = build_episodic_memory_context(prompt, messages)
    if not context:
        content = "這次查詢沒有找到足夠明確的對話事件。"
        metadata = {"hits": 0}
    else:
        content = context
        metadata = {"hits": context.count("[")}

    return ToolOutput(
        name="history_lookup",
        content=content,
        metadata=metadata,
    )
