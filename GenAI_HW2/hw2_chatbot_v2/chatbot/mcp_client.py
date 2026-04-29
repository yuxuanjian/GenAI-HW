import asyncio
import json
import os
import sys
from pathlib import Path

from .models import ToolOutput


BASE_DIR = Path(__file__).resolve().parent.parent


def _build_server_env() -> dict[str, str]:
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    base_dir_text = str(BASE_DIR)
    env["PYTHONPATH"] = (
        f"{base_dir_text}{os.pathsep}{python_path}" if python_path else base_dir_text
    )
    return env


def _extract_structured_payload(result) -> dict | None:
    for attribute in ("structuredContent", "structured_content"):
        payload = getattr(result, attribute, None)
        if isinstance(payload, dict):
            return payload
    return None


def _extract_text_blocks(result) -> list[str]:
    texts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return texts


def _parse_tool_result(tool_name: str, result) -> ToolOutput:
    payload = _extract_structured_payload(result)
    if isinstance(payload, dict):
        metadata = dict(payload.get("metadata", {}))
        metadata.setdefault("transport", "mcp-stdio")
        return ToolOutput(
            name=payload.get("name", tool_name),
            content=payload.get("content", ""),
            metadata=metadata,
        )

    text_blocks = _extract_text_blocks(result)
    if len(text_blocks) == 1:
        try:
            payload = json.loads(text_blocks[0])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            metadata = dict(payload.get("metadata", {}))
            metadata.setdefault("transport", "mcp-stdio")
            return ToolOutput(
                name=payload.get("name", tool_name),
                content=payload.get("content", ""),
                metadata=metadata,
            )

    return ToolOutput(
        name=tool_name,
        content="\n".join(text_blocks),
        metadata={"success": not getattr(result, "isError", False), "transport": "mcp-stdio"},
    )


def _build_tool_arguments(tool_name: str, prompt: str, chat_id: str) -> dict:
    if tool_name in {"document_retrieval", "history_lookup"}:
        return {"chat_id": chat_id, "query": prompt}
    return {"query": prompt}


async def _call_tools_via_mcp_async(prompt: str, tool_names: list[str], chat_id: str) -> list[ToolOutput]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    if not tool_names:
        return []

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "chatbot.mcp_server"],
        env=_build_server_env(),
    )

    outputs: list[ToolOutput] = []
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for tool_name in tool_names:
                result = await session.call_tool(
                    tool_name,
                    arguments=_build_tool_arguments(tool_name, prompt, chat_id),
                )
                outputs.append(_parse_tool_result(tool_name, result))

    return outputs


def call_tools_via_mcp(prompt: str, tool_names: list[str], chat_id: str) -> list[ToolOutput]:
    return asyncio.run(_call_tools_via_mcp_async(prompt, tool_names, chat_id))
