import math
import os
import re
from collections import Counter

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..config import DOCUMENT_RETRIEVAL_TOP_K
from ..models import RetrievalHit, ToolOutput


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")
CJK_PATTERN = re.compile(r"^[\u4e00-\u9fff]+$")
SUMMARY_HINT_KEYWORDS = (
    "整理",
    "統整",
    "總結",
    "摘要",
    "重點",
    "summarize",
    "summary",
    "overview",
    "paper",
    "論文",
    "文章",
)
BM25_K1 = 1.5
BM25_B = 0.75
EMBEDDING_CANDIDATE_MULTIPLIER = 6
EMBEDDING_TEXT_LIMIT = 3000


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0).lower()
        if CJK_PATTERN.fullmatch(token):
            if len(token) <= 2:
                tokens.append(token)
                continue
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
            tokens.extend(token[index : index + 3] for index in range(len(token) - 2))
            if len(token) <= 6:
                tokens.append(token)
            continue
        tokens.append(token)
    return tokens


def normalize_for_phrase_match(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def build_chunk_entries(document_library: dict) -> list[dict]:
    entries: list[dict] = []
    for doc_id, document in document_library.items():
        for chunk in document.get("chunks", []):
            content = chunk.get("content", "")
            tokens = tokenize(content)
            if not tokens:
                continue
            entries.append(
                {
                    "doc_id": doc_id,
                    "doc_name": document.get("name", "unknown"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "content": content,
                    "tokens": tokens,
                }
            )
    return entries


def build_document_frequency(entries: list[dict]) -> Counter:
    document_frequency: Counter = Counter()
    for entry in entries:
        document_frequency.update(set(entry["tokens"]))
    return document_frequency


def score_bm25(
    query_tokens: list[str],
    chunk_tokens: list[str],
    document_frequency: Counter,
    document_count: int,
    average_document_length: float,
) -> float:
    if not query_tokens or not chunk_tokens or document_count <= 0:
        return 0.0

    chunk_counter = Counter(chunk_tokens)
    chunk_length = len(chunk_tokens)
    score = 0.0
    for token in set(query_tokens):
        term_frequency = chunk_counter.get(token, 0)
        if term_frequency <= 0:
            continue

        frequency = document_frequency.get(token, 0)
        inverse_document_frequency = math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
        denominator = term_frequency + BM25_K1 * (1 - BM25_B + BM25_B * chunk_length / max(average_document_length, 1))
        score += inverse_document_frequency * (term_frequency * (BM25_K1 + 1)) / denominator
    return score


def score_phrase_bonus(query: str, content: str) -> float:
    normalized_query = normalize_for_phrase_match(query)
    normalized_content = normalize_for_phrase_match(content)
    if not normalized_query or not normalized_content:
        return 0.0

    bonus = 0.0
    if normalized_query in normalized_content:
        bonus += 8.0

    for token in set(tokenize(query)):
        if len(token) >= 3 and token in normalized_content:
            bonus += min(len(token) * 0.5, 3.0)
    return bonus


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def build_embedding_client():
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_BASE_URL")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if OpenAI is None or not api_key or not base_url or not embedding_model:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedding_model = os.getenv("EMBEDDING_MODEL")
    client = build_embedding_client()
    if not client or not embedding_model:
        raise RuntimeError("EMBEDDING_MODEL / CUSTOM_API_KEY / CUSTOM_BASE_URL 未完整設定。")

    response = client.embeddings.create(model=embedding_model, input=texts)
    return [item.embedding for item in response.data]


def rerank_with_embeddings(query: str, hits: list[RetrievalHit]) -> tuple[list[RetrievalHit], str]:
    if not os.getenv("EMBEDDING_MODEL"):
        return hits, "not_configured"
    if not hits:
        return hits, "no_candidates"

    try:
        texts = [query, *[hit.content[:EMBEDDING_TEXT_LIMIT] for hit in hits]]
        embeddings = embed_texts(texts)
    except Exception as error:
        return hits, f"failed: {error}"

    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]
    for hit, embedding in zip(hits, chunk_embeddings):
        hit.score = hit.score + max(cosine_similarity(query_embedding, embedding), 0.0) * 50

    hits.sort(key=lambda item: item.score, reverse=True)
    return hits, "embedding_rerank"


def is_summary_like_query(query: str) -> bool:
    normalized = query.lower()
    return any(keyword.lower() in normalized for keyword in SUMMARY_HINT_KEYWORDS)


def select_target_documents(query: str, document_library: dict) -> list[tuple[str, dict]]:
    if not document_library:
        return []

    items = list(document_library.items())
    if len(items) == 1:
        return items

    normalized = query.lower()
    matched: list[tuple[str, dict]] = []
    for doc_id, document in items:
        name = document.get("name", "")
        file_stem = name.rsplit(".", maxsplit=1)[0].lower()
        if file_stem and file_stem in normalized:
            matched.append((doc_id, document))

    return matched or items


def retrieve_relevant_chunks(query: str, document_library: dict, top_k: int = DOCUMENT_RETRIEVAL_TOP_K) -> tuple[list[RetrievalHit], str]:
    query_tokens = tokenize(query)
    entries = build_chunk_entries(document_library)
    if not query_tokens or not entries:
        return [], "bm25"

    document_frequency = build_document_frequency(entries)
    document_count = len(entries)
    average_document_length = sum(len(entry["tokens"]) for entry in entries) / max(document_count, 1)
    hits: list[RetrievalHit] = []

    for entry in entries:
        score = score_bm25(
            query_tokens=query_tokens,
            chunk_tokens=entry["tokens"],
            document_frequency=document_frequency,
            document_count=document_count,
            average_document_length=average_document_length,
        )
        score += score_phrase_bonus(query, entry["content"])
        if score <= 0:
            continue
        hits.append(
            RetrievalHit(
                doc_id=entry["doc_id"],
                doc_name=entry["doc_name"],
                chunk_index=entry["chunk_index"],
                content=entry["content"],
                score=score,
            )
        )

    hits.sort(key=lambda item: item.score, reverse=True)
    candidate_limit = max(top_k * EMBEDDING_CANDIDATE_MULTIPLIER, top_k)
    candidate_hits = hits[:candidate_limit]
    reranked_hits, embedding_status = rerank_with_embeddings(query, candidate_hits)
    return reranked_hits[:top_k], embedding_status


def build_representative_hits(target_documents: list[tuple[str, dict]], top_k: int) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []

    for doc_id, document in target_documents:
        chunks = document.get("chunks", [])
        if not chunks:
            continue

        if len(chunks) <= top_k:
            selected_indices = list(range(len(chunks)))
        else:
            selected_indices = [0, 1, len(chunks) // 2, len(chunks) - 1]

        seen_indices: set[int] = set()
        for chunk_index in selected_indices:
            if chunk_index in seen_indices or chunk_index >= len(chunks):
                continue
            seen_indices.add(chunk_index)
            chunk = chunks[chunk_index]
            hits.append(
                RetrievalHit(
                    doc_id=doc_id,
                    doc_name=document.get("name", "unknown"),
                    chunk_index=chunk.get("chunk_index", chunk_index),
                    content=chunk.get("content", ""),
                    score=0.01,
                )
            )

    return hits[:top_k]


def merge_unique_hits(primary_hits: list[RetrievalHit], extra_hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
    merged: list[RetrievalHit] = []
    seen_keys: set[tuple[str, int]] = set()

    for hit in [*primary_hits, *extra_hits]:
        key = (hit.doc_id, hit.chunk_index)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(hit)
        if len(merged) >= top_k:
            break

    return merged


def build_document_retrieval_tool_output(query: str, document_library: dict) -> ToolOutput:
    if not document_library:
        return ToolOutput(
            name="document_retrieval",
            content="目前知識庫裡沒有任何文件，因此無法進行文件檢索。",
            metadata={"hit_count": 0, "strategy": "no_documents"},
        )

    target_documents = select_target_documents(query, document_library=document_library)
    target_library = {doc_id: document for doc_id, document in target_documents}
    hits, embedding_status = retrieve_relevant_chunks(query, document_library=target_library)
    strategy = "bm25_embedding_rerank" if embedding_status == "embedding_rerank" else "bm25"
    summary_like = is_summary_like_query(query)

    if summary_like and target_documents:
        representative_hits = build_representative_hits(target_documents, top_k=DOCUMENT_RETRIEVAL_TOP_K)
        hits = merge_unique_hits(hits, representative_hits, top_k=DOCUMENT_RETRIEVAL_TOP_K)
        if hits:
            strategy = f"{strategy}_summary_mix"

    if not hits and target_documents:
        strategy = "summary_fallback" if summary_like else "document_fallback"
        hits = build_representative_hits(target_documents, top_k=DOCUMENT_RETRIEVAL_TOP_K)

    if not hits:
        return ToolOutput(
            name="document_retrieval",
            content="知識庫中沒有找到足夠相關的文件片段。",
            metadata={"hit_count": 0, "strategy": "no_hits"},
        )

    blocks = []
    for hit in hits:
        blocks.append(
            f"[{hit.doc_name} / chunk {hit.chunk_index + 1} / score {hit.score:.2f}]\n{hit.content}"
        )

    intro = "【已檢索到的文件片段】"
    if "embedding_rerank" in strategy:
        intro = "【RAG 檢索：BM25 初篩 + embedding rerank，已取回最相關文件片段】"
    elif strategy.startswith("bm25"):
        intro = "【RAG 檢索：BM25 檢索，已取回最相關文件片段】"

    if "summary_mix" in strategy:
        intro = "【文件摘要模式：系統混合了關鍵字命中與代表性片段】"
    elif strategy == "summary_fallback":
        intro = "【文件摘要模式：由於問題較像整篇統整，系統改抓代表性片段】"
    elif strategy == "document_fallback":
        intro = "【文件 fallback 模式：雖然詞面匹配較低，但系統仍提供目標文件片段】"

    return ToolOutput(
        name="document_retrieval",
        content=intro + "\n\n" + "\n\n".join(blocks),
        metadata={
            "hit_count": len(hits),
            "documents": [hit.doc_name for hit in hits],
            "strategy": strategy,
            "embedding_status": embedding_status,
        },
    )
