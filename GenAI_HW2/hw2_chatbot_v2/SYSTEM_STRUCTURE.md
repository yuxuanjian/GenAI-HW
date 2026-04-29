# System Structure
```text
Streamlit UI
  -> Thinking Status: show high-level progress outside final answer
  -> Router: choose qwen35-4b / qwen35-397b / vision model and required tools
  -> Tool Layer: MCP stdio client -> MCP server -> tools; local fallback if needed
  -> Context Builder: history + memory + tool results + image data
  -> OpenAI-compatible LLM Gateway: final response
```
| Component | Responsibility |
| --- | --- |
| UI | Chat/folder management, document upload, image attachment, model settings, MCP status, memory panel |
| Router | Selects model and tools; caption shows `Model`, `Tools`, `Route` |
| MCP Tools | Provides calculator, web_search, document_retrieval, history_lookup through stdio MCP |
| RAG | Chunks uploaded files; retrieves by BM25 with optional embedding rerank |
| Web Search | Manual Search toggle; Serper first, DuckDuckGo/Bing fallback, page/PDF excerpts |
| Vision + Query Planning | Vision model reads images; image/document clues can rewrite search queries before web search |
| Long-Term Memory | Uses pinned memory, per-chat summary, and structured JSON memory for cross-chat context |
## Flow
1. User sends text/image or asks about uploaded documents.
2. Router selects model/tools; Thinking Status shows progress without entering final answer.
3. Tools run through MCP first, with local fallback.
4. Prompt builder injects recent history, memory, tool results, and image data.
5. LLM returns answer; chat and memory are saved.
## Storage
```text
data/chat_storage.json        chats, messages, document index, per-chat memory
data/structured_memory.json   cross-chat structured memory
data/uploads/                 uploaded documents
data/attachments/             image attachments
```