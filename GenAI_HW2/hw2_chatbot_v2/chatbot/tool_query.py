import json
import re

from .models import ToolOutput
from .prompts import image_path_to_data_url


MAX_DOCUMENT_CONTEXT_CHARS = 2200
MAX_AUGMENTED_QUERY_CHARS = 220
MAX_CANDIDATE_QUERIES = 3
SEARCH_TERM_PATTERN = re.compile(r"@[A-Za-z0-9_]+|[A-Za-z][A-Za-z0-9_-]{2,}|[\u3040-\u30ff]{2,}|[\u4e00-\u9fff]{2,12}")
GENERIC_SEARCH_TERMS = {
    "圖片",
    "照片",
    "角色",
    "女性",
    "女生",
    "可能",
    "搜尋",
    "線索",
    "外觀",
    "作品",
    "動畫",
    "動漫",
    "遊戲",
    "插畫",
    "水印",
    "簽名",
    "文字",
    "左上角",
    "右上角",
    "左下角",
    "右下角",
    "左上角文字",
    "右上角文字",
    "左下角文字",
    "右下角文字",
    "頭髮",
    "眼睛",
    "制服",
    "表情",
}


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return {}

    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split())[:MAX_AUGMENTED_QUERY_CHARS]


def as_string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    strings: list[str] = []
    for item in value:
        text = normalize_query(str(item))
        if text:
            strings.append(text)
    return strings


def extract_search_terms(text: str, limit: int = 10) -> list[str]:
    terms: list[str] = []
    for match in SEARCH_TERM_PATTERN.finditer(text):
        term = match.group(0).strip()
        normalized_term = term.lower()
        if not term or normalized_term in GENERIC_SEARCH_TERMS:
            continue
        if len(term) <= 1:
            continue
        if term not in terms:
            terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def build_candidate_queries(
    original_query: str,
    augmented_query: str,
    visual_summary: str = "",
    document_summary: str = "",
    alternative_queries: list[str] | None = None,
) -> list[str]:
    candidates: list[str] = []

    for query in [augmented_query, *(alternative_queries or [])]:
        normalized = normalize_query(query)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    visual_terms = extract_search_terms(visual_summary, limit=8)
    document_terms = extract_search_terms(document_summary, limit=6)
    extracted_terms = visual_terms or document_terms
    if extracted_terms:
        term_query = normalize_query(" ".join(extracted_terms))
        if term_query and term_query not in candidates:
            candidates.append(term_query)

        enriched_query = normalize_query(f"{augmented_query} {' '.join(extracted_terms[:5])}")
        if enriched_query and enriched_query not in candidates:
            candidates.insert(0, enriched_query)

    fallback_query = normalize_query(original_query)
    if fallback_query and fallback_query not in candidates and not extracted_terms:
        candidates.append(fallback_query)

    return candidates[:MAX_CANDIDATE_QUERIES]


def build_query_planning_output(
    original_query: str,
    augmented_query: str,
    reason: str,
    visual_summary: str = "",
    document_summary: str = "",
    candidate_queries: list[str] | None = None,
    confidence: float | None = None,
) -> ToolOutput:
    confidence_text = f"\n信心: {confidence:.2f}" if isinstance(confidence, (int, float)) else ""
    visual_text = f"\n圖片線索: {visual_summary}" if visual_summary else ""
    document_text = f"\n文件線索: {document_summary}" if document_summary else ""
    candidates = candidate_queries or [augmented_query]
    candidate_text = "\n候選搜尋查詢:\n" + "\n".join(f"- {query}" for query in candidates)
    content = (
        "【工具查詢規劃】\n"
        f"原始問題: {original_query}\n"
        f"強化搜尋查詢: {augmented_query}\n"
        f"原因: {reason}"
        f"{visual_text}"
        f"{document_text}"
        f"{candidate_text}"
        f"{confidence_text}"
    )
    return ToolOutput(
        name="query_planning",
        content=content,
        metadata={
            "original_query": original_query,
            "augmented_query": augmented_query,
            "candidate_queries": candidates,
            "visual_summary": visual_summary,
            "document_summary": document_summary,
            "confidence": confidence,
        },
    )


def build_failed_query_planning_output(original_query: str, error: Exception) -> ToolOutput:
    return ToolOutput(
        name="query_planning",
        content=(
            "【工具查詢規劃】\n"
            f"原始問題: {original_query}\n"
            "強化搜尋查詢: （失敗，改用原始問題）\n"
            f"原因: {error}"
        ),
        metadata={"original_query": original_query, "augmented_query": original_query, "success": False},
    )


def build_search_planning_messages(
    user_query: str,
    image_attachment: dict | None,
    document_context: str,
) -> list[dict]:
    system_message = (
        "你是 web search query planner。"
        "請根據使用者問題、圖片線索與文件片段，產生最適合搜尋引擎使用的查詢。"
        "重點是抽出可搜尋的實體名稱、浮水印、作者名、OCR 文字、作品名、英文別名、關鍵術語。"
        "如果有圖片，必須先列出看見的精確文字、@handle、簽名、水印、角色外觀；不要只寫「動漫角色」。"
        "搜尋查詢必須包含圖片中的精確文字或最有辨識度的外觀線索。"
        "不要回答問題，只輸出 JSON。"
    )
    user_text = (
        f"使用者問題：{user_query}\n\n"
        f"文件片段：\n{document_context[:MAX_DOCUMENT_CONTEXT_CHARS] if document_context else '（無）'}\n\n"
        "請輸出 JSON：\n"
        "{\n"
        '  "query": "搜尋引擎查詢，最多 20 個詞",\n'
        '  "alternative_queries": ["其他可嘗試搜尋查詢，最多 2 筆"],\n'
        '  "visual_summary": "如果有圖片，列出看見的文字、簽名、角色外觀；沒有圖片則空字串",\n'
        '  "document_summary": "如果有文件，列出可用於搜尋的文件線索；沒有文件則空字串",\n'
        '  "reason": "為何這樣搜尋",\n'
        '  "confidence": 0.0\n'
        "}\n"
        "查詢要保留原問題意圖。例如問角色身份時，查詢應包含圖片中的文字/畫師/角色外觀；問文件外部資訊時，查詢應包含文件中的專有名詞。"
    )

    if not image_attachment:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_text},
        ]

    data_url = image_path_to_data_url(image_attachment.get("path", ""))
    content_parts = [{"type": "text", "text": user_text}]
    if data_url:
        content_parts.append({"type": "image_url", "image_url": {"url": data_url}})

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": content_parts},
    ]


def plan_augmented_search_query(
    client,
    model: str,
    user_query: str,
    image_attachment: dict | None = None,
    document_context: str = "",
) -> tuple[str, ToolOutput | None]:
    from .llm import complete_text

    messages = build_search_planning_messages(
        user_query=user_query,
        image_attachment=image_attachment,
        document_context=document_context,
    )
    raw_response = complete_text(
        client=client,
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=500,
    )
    parsed = extract_json_object(raw_response)
    augmented_query = normalize_query(str(parsed.get("query", "")))
    if not augmented_query:
        return user_query, None
    visual_summary = str(parsed.get("visual_summary", "") or "")
    document_summary = str(parsed.get("document_summary", "") or "")
    alternative_queries = as_string_list(parsed.get("alternative_queries"))
    candidate_queries = build_candidate_queries(
        original_query=user_query,
        augmented_query=augmented_query,
        visual_summary=visual_summary,
        document_summary=document_summary,
        alternative_queries=alternative_queries,
    )

    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None

    output = build_query_planning_output(
        original_query=user_query,
        augmented_query=augmented_query,
        reason=str(parsed.get("reason", "") or "根據圖片/文件線索強化搜尋查詢。"),
        visual_summary=visual_summary,
        document_summary=document_summary,
        candidate_queries=candidate_queries,
        confidence=confidence,
    )
    return candidate_queries[0], output


def extract_document_context(tool_output: ToolOutput | None) -> str:
    if not tool_output or tool_output.name != "document_retrieval":
        return ""
    return tool_output.content[:MAX_DOCUMENT_CONTEXT_CHARS]
