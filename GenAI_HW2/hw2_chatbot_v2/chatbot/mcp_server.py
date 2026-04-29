import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .tools.calculator import build_calculator_tool_output
from .tools.history_lookup import build_history_lookup_tool_output
from .tools.retrieval import build_document_retrieval_tool_output
from .tools.web_search import build_web_search_tool_output


mcp = FastMCP("HW2 Tools MCP Server")
SAVE_FILE = Path(__file__).resolve().parent.parent / "data" / "chat_storage.json"


def load_payload() -> dict:
    if not SAVE_FILE.exists():
        return {"chats": {}}

    try:
        return json.loads(SAVE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {"chats": {}}


def load_chat_record(chat_id: str) -> dict | None:
    payload = load_payload()
    return payload.get("chats", {}).get(chat_id)


def tool_output_to_dict(tool_output) -> dict:
    return {
        "name": tool_output.name,
        "content": tool_output.content,
        "metadata": tool_output.metadata,
    }


def build_missing_chat_output(tool_name: str, chat_id: str) -> dict:
    return {
        "name": tool_name,
        "content": f"找不到 chat_id={chat_id} 的對話資料，因此無法執行 {tool_name}。",
        "metadata": {"success": False, "chat_id": chat_id},
    }


def drop_current_query(messages: list[dict], query: str) -> list[dict]:
    if not messages:
        return []

    last_message = messages[-1]
    if last_message.get("role") == "user" and (last_message.get("content") or "").strip() == query.strip():
        return messages[:-1]
    return messages


@mcp.tool()
def calculator(query: str) -> dict:
    return tool_output_to_dict(build_calculator_tool_output(query))


@mcp.tool()
def web_search(query: str) -> dict:
    return tool_output_to_dict(build_web_search_tool_output(query))


@mcp.tool()
def document_retrieval(chat_id: str, query: str) -> dict:
    chat_record = load_chat_record(chat_id)
    if not chat_record:
        return build_missing_chat_output("document_retrieval", chat_id)

    return tool_output_to_dict(
        build_document_retrieval_tool_output(
            query,
            document_library=chat_record.get("document_library", {}),
        )
    )


@mcp.tool()
def history_lookup(chat_id: str, query: str) -> dict:
    chat_record = load_chat_record(chat_id)
    if not chat_record:
        return build_missing_chat_output("history_lookup", chat_id)

    messages = drop_current_query(chat_record.get("messages", []), query)
    return tool_output_to_dict(build_history_lookup_tool_output(query, messages))


@mcp.resource("chat://{chat_id}/memory-summary")
def read_memory_summary(chat_id: str) -> str:
    chat_record = load_chat_record(chat_id)
    if not chat_record:
        return ""
    return chat_record.get("memory_summary", "")


@mcp.resource("chat://{chat_id}/documents")
def read_document_index(chat_id: str) -> str:
    chat_record = load_chat_record(chat_id)
    if not chat_record:
        return "[]"

    documents = []
    for doc_id, document in chat_record.get("document_library", {}).items():
        documents.append(
            {
                "doc_id": doc_id,
                "name": document.get("name", ""),
                "file_type": document.get("file_type", ""),
                "chunk_count": document.get("chunk_count", 0),
                "uploaded_at": document.get("uploaded_at", ""),
            }
        )
    return json.dumps(documents, ensure_ascii=False, indent=2)


@mcp.resource("chat://{chat_id}/messages")
def read_chat_messages(chat_id: str) -> str:
    chat_record = load_chat_record(chat_id)
    if not chat_record:
        return "[]"
    return json.dumps(chat_record.get("messages", []), ensure_ascii=False, indent=2)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
