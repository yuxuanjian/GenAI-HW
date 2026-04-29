import os
import re
from datetime import datetime
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

try:
    import fitz
except ImportError:
    fitz = None

from ..config import WEB_SEARCH_MAX_RESULTS
from ..models import ToolOutput


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
OFFICIAL_DOMAINS = ("openai.com", "help.openai.com")
OFFICIAL_SEARCH_TERMS = ("openai", "chatgpt", "gpt")
FRESHNESS_KEYWORDS = ("最新", "今天", "近期", "目前", "現在", "進展", "current", "latest", "recent", "today", "news")
MAX_PDF_BYTES = 12 * 1024 * 1024
PDF_TEXT_MAX_PAGES = 24
QUERY_KEYWORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|\d{2,}|第\s*\d+\s*[週周]?|[\u4e00-\u9fff]{2,}")
QUERY_FILLER_PATTERN = re.compile(
    r"^(?:請|麻煩|幫我|幫忙|可以)?\s*(?:搜尋|查詢|查|找|google|上網查)\s*",
    re.IGNORECASE,
)
MAX_SERPER_CANDIDATES = 10


def build_search_query(query: str) -> str:
    current_year = str(datetime.now().year)
    normalized = query.lower()
    has_year = any(str(year) in query for year in range(2024, 2031))
    if is_freshness_query(normalized) and not has_year:
        return f"{query} {current_year}"
    return query


def is_freshness_query(query: str) -> bool:
    normalized = query.lower()
    return any(keyword in normalized for keyword in FRESHNESS_KEYWORDS)


def parse_serper_items(items: list[dict]) -> list[dict]:
    results = []
    for item in items:
        snippet_parts = []
        source = item.get("source")
        date = item.get("date")
        snippet = item.get("snippet", "")
        if source:
            snippet_parts.append(f"來源: {source}")
        if date:
            snippet_parts.append(f"日期: {date}")
        if snippet:
            snippet_parts.append(snippet)

        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": " | ".join(snippet_parts),
            }
        )
    return results


def parse_serper_payload(payload: dict, max_results: int) -> list[dict]:
    results: list[dict] = []

    answer_box = payload.get("answerBox") or {}
    answer_link = answer_box.get("link") or answer_box.get("sourceLink")
    answer_text = answer_box.get("answer") or answer_box.get("snippet") or answer_box.get("snippetHighlighted")
    if answer_text:
        results.append(
            {
                "title": answer_box.get("title") or "Answer Box",
                "url": answer_link or "https://google.com/search",
                "snippet": f"精選摘要: {answer_text}",
            }
        )

    knowledge_graph = payload.get("knowledgeGraph") or {}
    knowledge_link = knowledge_graph.get("website") or knowledge_graph.get("sourceLink")
    knowledge_description = knowledge_graph.get("description")
    if knowledge_description:
        results.append(
            {
                "title": knowledge_graph.get("title") or "Knowledge Graph",
                "url": knowledge_link or "https://google.com/search",
                "snippet": f"知識圖譜: {knowledge_description}",
            }
        )

    results.extend(parse_serper_items(payload.get("organic", [])[:max_results]))
    return results


def search_single_serper_query(query: str, max_results: int) -> tuple[str, list[dict]]:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY 未設定。")

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    request_payload = {"q": query, "num": max_results, "gl": "tw", "hl": "zh-tw"}
    results = []
    provider_parts = ["serper"]
    errors = []

    if is_freshness_query(query):
        try:
            news_response = requests.post(
                "https://google.serper.dev/news",
                headers=headers,
                json=request_payload,
                timeout=20,
            )
            news_response.raise_for_status()
            results.extend(parse_serper_items(news_response.json().get("news", [])[:max_results]))
            provider_parts.append("news")
        except Exception as error:
            errors.append(f"news: {error}")

    try:
        search_response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=request_payload,
            timeout=20,
        )
        search_response.raise_for_status()
        results.extend(parse_serper_payload(search_response.json(), max_results=max_results))
        provider_parts.append("search")
    except Exception as error:
        errors.append(f"search: {error}")

    if not results and errors:
        raise RuntimeError("; ".join(errors))

    return "+".join(provider_parts), results


def search_with_serper(query: str, max_results: int) -> tuple[str, list[dict]]:
    candidate_limit = max(max_results * 2, MAX_SERPER_CANDIDATES)
    merged_results: list[dict] = []
    provider_names: list[str] = []
    used_queries: list[str] = []
    errors: list[str] = []
    fallback_queries = build_fallback_queries(query)
    search_queries = [query, *fallback_queries]
    minimum_query_attempts = min(len(search_queries), 2 if fallback_queries else 1)

    for query_index, search_query in enumerate(search_queries):
        try:
            provider_name, results = search_single_serper_query(search_query, max_results=candidate_limit)
        except Exception as error:
            errors.append(f"{search_query}: {error}")
            continue

        provider_names.append(provider_name)
        used_queries.append(search_query)
        merged_results = merge_unique_results(merged_results, clean_results(results))
        if len(merged_results) >= candidate_limit and query_index + 1 >= minimum_query_attempts:
            break

    if not merged_results and errors:
        raise RuntimeError("; ".join(errors))

    provider = "+".join(dict.fromkeys(provider_names)) or "serper"
    if used_queries:
        provider = f"{provider}:{' | '.join(used_queries[:3])}"
    return provider, merged_results[:candidate_limit]


def search_with_duckduckgo(query: str, max_results: int) -> tuple[str, list[dict]]:
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for result in soup.select(".result"):
        anchor = result.select_one(".result__a") or result.select_one(".result__title a")
        snippet = result.select_one(".result__snippet")
        if not anchor:
            continue

        href = anchor.get("href", "")
        if href.startswith("//"):
            href = f"https:{href}"

        results.append(
            {
                "title": anchor.get_text(" ", strip=True),
                "url": href,
                "snippet": snippet.get_text(" ", strip=True) if snippet else "",
            }
        )
        if len(results) >= max_results:
            break

    return "duckduckgo", results


def search_with_bing(query: str, max_results: int) -> tuple[str, list[dict]]:
    response = requests.get(
        "https://www.bing.com/search",
        params={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for result in soup.select("li.b_algo"):
        anchor = result.select_one("h2 a")
        snippet = result.select_one(".b_caption p") or result.select_one("p")
        if not anchor:
            continue

        results.append(
            {
                "title": anchor.get_text(" ", strip=True),
                "url": anchor.get("href", ""),
                "snippet": snippet.get_text(" ", strip=True) if snippet else "",
            }
        )
        if len(results) >= max_results:
            break

    return "bing", results


def should_prefer_official_sources(query: str) -> bool:
    normalized = query.lower()
    return any(term in normalized for term in OFFICIAL_SEARCH_TERMS)


def preferred_domains_for_query(query: str) -> tuple[str, ...]:
    normalized = query.lower()
    domains: list[str] = []
    if should_prefer_official_sources(query):
        domains.extend(OFFICIAL_DOMAINS)
    if any(term in normalized for term in ("nycu", "陽明交通", "交大", "行事曆")):
        domains.extend(("nycu.edu.tw", "aa.nycu.edu.tw"))
    if any(term in normalized for term in ("reuters", "ap ", "iran", "伊朗", "美伊")):
        domains.extend(("reuters.com", "apnews.com", "axios.com"))
    return tuple(dict.fromkeys(domains))


def should_prefer_recent_results(query: str) -> bool:
    normalized = query.lower()
    return is_freshness_query(normalized) or any(
        term in normalized
        for term in ("行事曆", "calendar", "修訂", "114-2", "115年", "進展", "談判", "latest", "current")
    )


def extract_result_date_score(result: dict) -> int:
    text = f"{result.get('title', '')} {result.get('snippet', '')}"
    best_score = 0
    patterns = (
        r"(?P<year>\d{4})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})",
        r"(?P<year>\d{3,4})\s*年\s*(?P<month>\d{1,2})\s*(?:月)?\s*(?P<day>\d{1,2})\s*(?:日)?",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                year = int(match.group("year"))
                month = int(match.group("month"))
                day = int(match.group("day"))
            except (TypeError, ValueError):
                continue
            if year < 1911:
                year += 1911
            if not (1 <= month <= 12 and 1 <= day <= 31):
                continue
            best_score = max(best_score, year * 10000 + month * 100 + day)
    return best_score


def normalize_result_url(url: str) -> tuple[str, bool]:
    if not url:
        return "", False

    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    is_ad = "ad_domain" in query_params or parsed.path.endswith("/y.js")

    if "uddg" in query_params and query_params["uddg"]:
        return unquote(query_params["uddg"][0]), is_ad

    return url, is_ad


def domain_from_url(url: str) -> str:
    hostname = urlparse(url).hostname or ""
    return hostname.lower()


def is_official_domain(url: str) -> bool:
    domain = domain_from_url(url)
    return any(domain == official or domain.endswith(f".{official}") for official in OFFICIAL_DOMAINS)


def prioritize_results(query: str, results: list[dict]) -> list[dict]:
    if not results:
        return []

    preferred_domains = preferred_domains_for_query(query)
    prefer_recent = should_prefer_recent_results(query)

    annotated = []
    for index, result in enumerate(results):
        url = result.get("url", "")
        domain = domain_from_url(url)
        domain_bonus = 0
        for preferred_domain in preferred_domains:
            if domain == preferred_domain or domain.endswith(f".{preferred_domain}"):
                domain_bonus = 1
                break
        date_score = extract_result_date_score(result) if prefer_recent else 0
        annotated.append((domain_bonus, date_score, -index, result))

    annotated.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [result for _, _, _, result in annotated]


def clean_results(results: list[dict]) -> list[dict]:
    cleaned_results = []
    seen_urls: set[str] = set()

    for result in results:
        normalized_url, is_ad = normalize_result_url(result.get("url", ""))
        if is_ad or not normalized_url or normalized_url in seen_urls:
            continue

        seen_urls.add(normalized_url)
        cleaned_results.append(
            {
                **result,
                "url": normalized_url,
                "is_official": is_official_domain(normalized_url),
            }
        )

    return cleaned_results


def merge_unique_results(*result_groups: list[dict]) -> list[dict]:
    merged_results = []
    seen_urls: set[str] = set()

    for result_group in result_groups:
        for result in result_group:
            url = result.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            merged_results.append(result)

    return merged_results


def search_official_sources(query: str, max_results: int) -> list[dict]:
    if not should_prefer_official_sources(query):
        return []

    official_results: list[dict] = []
    for domain in OFFICIAL_DOMAINS:
        try:
            _, results = search_with_duckduckgo(f"{query} site:{domain}", max_results=max_results)
        except Exception:
            continue
        official_results.extend(results)

    return clean_results(official_results)


def clean_search_query(query: str) -> str:
    cleaned = normalize_whitespace(QUERY_FILLER_PATTERN.sub("", query))
    cleaned = re.sub(r"(?:是什麼|是甚麼|有哪些|如何|怎麼|幫我|請|整理|說明|一下)[？?。,.，]*$", "", cleaned)
    return cleaned or query


def build_fallback_queries(query: str) -> list[str]:
    current_year = str(datetime.now().year)
    cleaned_query = clean_search_query(query)
    queries: list[str] = []

    if cleaned_query != query:
        queries.append(cleaned_query)

    if "交大" in query or "陽明交通" in query or "nycu" in query.lower():
        expanded = cleaned_query.replace("交大", "陽明交通大學 NYCU")
        queries.extend(
            [
                expanded,
                f"{expanded} site:nycu.edu.tw",
                expanded.replace("114-2", "114學年度第二學期"),
                expanded.replace("114-2", "114學年度第2學期"),
            ]
        )

    if "美伊" in query:
        queries.extend(
            [
                cleaned_query.replace("美伊", "美國 伊朗"),
                cleaned_query.replace("美伊戰爭", "美國 伊朗 衝突"),
                f"US Iran negotiations latest {current_year}",
                f"United States Iran talks latest {current_year}",
            ]
        )

    if "台積電" in query:
        queries.extend(
            [
                cleaned_query.replace("台積電", "TSMC"),
                f"TSMC stock news today {current_year}",
                f"2330 Taiwan stock news today {current_year}",
            ]
        )

    if should_prefer_official_sources(query):
        hyphenated = re.sub(r"\bGPT(?=\d)", "GPT-", cleaned_query, flags=re.IGNORECASE)
        queries.extend(
            [
                hyphenated,
                f"{hyphenated} site:openai.com",
                f"{hyphenated} site:help.openai.com",
                f"OpenAI model news {current_year}",
            ]
        )

    unique_queries = []
    seen = {query}
    for fallback_query in queries:
        normalized = " ".join(fallback_query.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_queries.append(normalized)
    return unique_queries


def search_without_key(query: str, max_results: int) -> tuple[str, list[dict]]:
    for search_query in [query, *build_fallback_queries(query)]:
        for provider in (search_with_duckduckgo, search_with_bing):
            try:
                provider_name, results = provider(search_query, max_results=max_results)
            except Exception:
                continue

            cleaned_results = clean_results(results)
            if cleaned_results:
                return f"{provider_name}:{search_query}", cleaned_results[:max_results]

    return "duckduckgo/bing", []


def is_pdf_url_or_response(url: str, response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    path = urlparse(url).path.lower()
    return "application/pdf" in content_type or path.endswith(".pdf")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_text(pdf_bytes: bytes, max_pages: int = PDF_TEXT_MAX_PAGES) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF 未安裝，無法解析 PDF。")

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_texts = []
        for page_index in range(min(max_pages, document.page_count)):
            page = document.load_page(page_index)
            text = page.get_text("text").strip()
            if text:
                page_texts.append(text)
        return "\n".join(page_texts)
    finally:
        document.close()


def build_excerpt_keywords(query: str) -> list[str]:
    keywords: list[str] = []
    for match in QUERY_KEYWORD_PATTERN.finditer(query):
        keyword = normalize_whitespace(match.group(0))
        if keyword and keyword not in keywords:
            keywords.append(keyword)
        if "周" in keyword:
            alternate = keyword.replace("周", "週")
            if alternate not in keywords:
                keywords.append(alternate)
        if "週" in keyword:
            alternate = keyword.replace("週", "周")
            if alternate not in keywords:
                keywords.append(alternate)

    for compact_week in re.findall(r"第\s*(\d+)\s*[週周]?", query):
        for keyword in (f"第{compact_week}週", f"第{compact_week}周", f"{compact_week}週", f"{compact_week}周", compact_week):
            if keyword not in keywords:
                keywords.append(keyword)
    return keywords


def split_text_for_excerpt(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    normalized_text = normalize_whitespace(text)
    if not normalized_text:
        return []

    chunks = []
    start = 0
    while start < len(normalized_text):
        chunk = normalized_text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(normalized_text):
            break
        start += max(chunk_size - overlap, 1)
    return chunks


def score_excerpt_chunk(chunk: str, keywords: list[str]) -> int:
    normalized_chunk = chunk.lower()
    score = 0
    for keyword in keywords:
        normalized_keyword = keyword.lower()
        if not normalized_keyword:
            continue
        score += normalized_chunk.count(normalized_keyword) * max(len(normalized_keyword), 2)
    return score


def select_relevant_excerpt(text: str, query: str, max_chars: int) -> str:
    chunks = split_text_for_excerpt(text)
    if not chunks:
        return ""

    keywords = build_excerpt_keywords(query)
    if not keywords:
        return chunks[0][:max_chars].strip()

    best_chunk = max(chunks, key=lambda chunk: score_excerpt_chunk(chunk, keywords))
    if score_excerpt_chunk(best_chunk, keywords) <= 0:
        best_chunk = chunks[0]
    return best_chunk[:max_chars].strip()


def fetch_page_excerpt(url: str, query: str = "", max_chars: int = 900) -> str:
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        response.raise_for_status()
    except Exception:
        return ""

    if is_pdf_url_or_response(url, response):
        content_length = int(response.headers.get("Content-Length") or 0)
        if content_length > MAX_PDF_BYTES or len(response.content) > MAX_PDF_BYTES:
            return "（PDF 檔案過大，未解析內文）"
        try:
            pdf_text = extract_pdf_text(response.content)
        except Exception as error:
            return f"（PDF 文字解析失敗：{error}）"
        excerpt = select_relevant_excerpt(pdf_text, query, max_chars=max_chars)
        return f"（PDF文字摘錄）{excerpt}" if excerpt else "（PDF 未抽取到可用文字）"

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    texts = []
    for node in soup.select("article p, main p, p, li"):
        text = node.get_text(" ", strip=True)
        if text:
            texts.append(text)
        if sum(len(item) for item in texts) >= max_chars:
            break

    excerpt = " ".join(texts)
    return select_relevant_excerpt(excerpt, query, max_chars=max_chars)


def build_web_search_tool_output(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> ToolOutput:
    actual_query = build_search_query(query)
    official_results: list[dict] = []
    try:
        provider, results = search_with_serper(actual_query, max_results=max_results)
    except Exception:
        try:
            provider, results = search_without_key(actual_query, max_results=max_results)
            official_results = search_official_sources(actual_query, max_results=3)
        except Exception as error:
            return ToolOutput(
                name="web_search",
                content=f"web search 執行失敗：{error}",
                metadata={"success": False, "query": actual_query},
            )

    searched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cleaned_results = clean_results(results)

    if not cleaned_results:
        try:
            fallback_provider, fallback_results = search_without_key(actual_query, max_results=max_results)
            provider = f"{provider}; fallback={fallback_provider}"
            cleaned_results = clean_results(fallback_results)
        except Exception:
            cleaned_results = []

    if should_prefer_official_sources(actual_query):
        official_results = search_official_sources(actual_query, max_results=3)

    results = prioritize_results(query, merge_unique_results(official_results, cleaned_results))

    if not results:
        return ToolOutput(
            name="web_search",
            content=f"搜尋時間: {searched_at}\n搜尋查詢: {actual_query}\n沒有找到可用的搜尋結果。",
            metadata={"success": True, "provider": provider, "result_count": 0, "query": actual_query, "searched_at": searched_at},
        )

    blocks = []
    official_hit_count = 0
    for index, result in enumerate(results[:max_results], start=1):
        official_prefix = "[官方來源] " if result.get("is_official") else ""
        if result.get("is_official"):
            official_hit_count += 1
        should_fetch_excerpt = index <= 3 or urlparse(result["url"]).path.lower().endswith(".pdf")
        excerpt = fetch_page_excerpt(result["url"], query=query) if should_fetch_excerpt else ""
        blocks.append(
            f"[結果 {index}]\n標題: {official_prefix}{result['title']}\n連結: {result['url']}\n搜尋摘要: {result['snippet']}\n頁面摘錄: {excerpt or '（未成功抓取頁面內文）'}"
        )

    official_note = ""
    if should_prefer_official_sources(query):
        official_note = (
            f"官方來源命中數: {official_hit_count}\n"
            "判讀要求: OpenAI / GPT / ChatGPT 相關問題應優先採用官方來源；非官方來源只能作為補充。\n\n"
        )

    return ToolOutput(
        name="web_search",
        content=(
            f"【網路搜尋結果】\n"
            f"搜尋時間: {searched_at}\n"
            f"原始查詢: {query}\n"
            f"實際搜尋查詢: {actual_query}\n"
            f"提供者: {provider}\n\n"
            f"{official_note}"
            + "\n\n".join(blocks)
        ),
        metadata={
            "success": True,
            "provider": provider,
            "result_count": len(results[:max_results]),
            "official_hit_count": official_hit_count,
            "query": actual_query,
            "searched_at": searched_at,
        },
    )
