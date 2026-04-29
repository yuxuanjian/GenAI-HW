import re
from collections import Counter

from .config import EPISODIC_MEMORY_TOP_K


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
ORDINAL_PATTERN = re.compile(r"第\s*(\d+)\s*(?:個|則|次|句|條|段|筆|輪)?")
RANGE_PATTERN = re.compile(r"(?:最一開始|一開始|最早)?\s*的?\s*前\s*([一二兩三四五六七八九十\d]+)\s*(?:個|則|次|句|條|段|筆|輪)?")
EARLIEST_RANGE_PATTERN = re.compile(r"(?:最一開始|一開始|最早)\s*的?\s*([一二兩三四五六七八九十\d]+)\s*(?:個|則|次|句|條|段|筆|輪)")
RECENT_RANGE_PATTERN = re.compile(r"(?:最近|最後|上)\s*([一二兩三四五六七八九十\d]+)\s*(?:個|則|次|句|條|段|筆|輪)")
CHINESE_ORDINAL_MAP = {
    "第一": 1,
    "第二": 2,
    "第三": 3,
    "第四": 4,
    "第五": 5,
    "第六": 6,
    "第七": 7,
    "第八": 8,
    "第九": 9,
    "第十": 10,
}
CHINESE_NUMBER_MAP = {
    "一": 1,
    "二": 2,
    "兩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
CONFIRMATION_PREFIXES = (
    "你確定",
    "你是說",
    "所以",
    "那",
    "請問",
    "請確認",
    "確認一下",
    "確定",
)
CHINESE_ORDINAL_PHRASES = "|".join(re.escape(phrase) for phrase in CHINESE_ORDINAL_MAP)
CONFIRMATION_CANDIDATE_PATTERN = re.compile(
    rf"(?P<candidate>[^，。！？?\n]+?)是(?:(?:{CHINESE_ORDINAL_PHRASES}|第\s*\d+)(?:\s*(?:個|則|次|句|條|段|筆|輪)?)?)(?:\s*(?:prompt|指令|問題|任務|回答|回覆))",
    re.IGNORECASE,
)
EXPLICIT_HISTORY_MARKERS = (
    "之前",
    "前面",
    "先前",
    "剛剛",
    "剛才",
    "最早",
    "一開始",
    "還記得",
    "history",
    "earlier",
    "previous",
)
EXPLICIT_DIALOGUE_MARKERS = (
    "prompt",
    "指令",
    "問題",
    "任務",
    "對話",
    "訊息",
    "句話",
    "message",
    "回答",
    "回覆",
    "我說",
    "我問",
    "我剛",
    "我之前",
    "我前面",
    "我先前",
    "你說",
    "你回答",
    "你剛",
    "你之前",
    "助理",
    "使用者",
    "assistant",
    "user",
)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def parse_chinese_number(text: str) -> int | None:
    normalized = text.strip()
    if not normalized:
        return None
    if normalized.isdigit():
        return int(normalized)
    if normalized in CHINESE_NUMBER_MAP:
        return CHINESE_NUMBER_MAP[normalized]
    if "十" in normalized:
        left, _, right = normalized.partition("十")
        tens = CHINESE_NUMBER_MAP.get(left, 1) if left else 1
        ones = CHINESE_NUMBER_MAP.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def has_range_reference(query: str) -> bool:
    return bool(RANGE_PATTERN.search(query) or EARLIEST_RANGE_PATTERN.search(query) or RECENT_RANGE_PATTERN.search(query))


def is_history_query(query: str) -> bool:
    normalized = query.lower()
    if any(marker.lower() in normalized for marker in EXPLICIT_HISTORY_MARKERS):
        return True

    has_ordinal_reference = any(phrase in query for phrase in CHINESE_ORDINAL_MAP) or bool(ORDINAL_PATTERN.search(query))
    has_ordinal_reference = has_ordinal_reference or has_range_reference(query)
    has_dialogue_reference = any(marker.lower() in normalized for marker in EXPLICIT_DIALOGUE_MARKERS)
    return has_ordinal_reference and has_dialogue_reference


def infer_target_role(query: str) -> str:
    normalized = query.lower()
    user_markers = ("prompt", "指令", "我說", "我問", "教你", "使用者", "user")
    assistant_markers = ("你說", "你回答", "助理", "assistant", "回覆")

    if "你" in query and any(marker in query for marker in ("回答", "回覆", "說", "講", "提到")):
        return "assistant"
    if "我" in query and any(marker in query for marker in ("說", "問", "講", "提到", "下的指令")):
        return "user"

    if any(marker.lower() in normalized for marker in assistant_markers):
        return "assistant"
    if any(marker.lower() in normalized for marker in user_markers):
        return "user"
    return "user"


def extract_requested_turn_numbers(query: str, total_turns: int) -> list[int]:
    numbers: set[int] = set()
    normalized = query.lower()

    for pattern in (RANGE_PATTERN, EARLIEST_RANGE_PATTERN):
        for match in pattern.finditer(query):
            count = parse_chinese_number(match.group(1))
            if count:
                numbers.update(range(1, min(count, total_turns) + 1))

    for match in RECENT_RANGE_PATTERN.finditer(query):
        count = parse_chinese_number(match.group(1))
        if count and total_turns > 0:
            start = max(total_turns - count + 1, 1)
            numbers.update(range(start, total_turns + 1))

    for phrase, number in CHINESE_ORDINAL_MAP.items():
        if phrase in query:
            numbers.add(number)

    for match in ORDINAL_PATTERN.finditer(query):
        numbers.add(int(match.group(1)))

    if "最早" in query or "一開始" in query or "第一件" in query or "第一次" in query:
        numbers.add(1)

    if total_turns > 0 and ("最後" in query or "最近" in query):
        numbers.add(total_turns)

    if total_turns > 0 and ("上一個" in query or "上一則" in query or "上一次" in query):
        numbers.add(total_turns)

    return sorted(number for number in numbers if 1 <= number <= total_turns)


def build_message_events(messages: list[dict]) -> list[dict]:
    events: list[dict] = []
    user_turn = 0
    assistant_turn = 0

    for event_index, message in enumerate(messages, start=1):
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if not content:
            continue

        if role == "user":
            user_turn += 1
            role_turn = user_turn
        elif role == "assistant":
            assistant_turn += 1
            role_turn = assistant_turn
        else:
            role_turn = 0

        events.append(
            {
                "event_index": event_index,
                "role": role,
                "role_turn": role_turn,
                "content": content,
                "attachments": message.get("attachments", []),
            }
        )

    return events


def collect_requested_events(query: str, messages: list[dict]) -> tuple[str, list[dict], list[dict]]:
    events = build_message_events(messages)
    if not events:
        return "user", [], []

    target_role = infer_target_role(query)
    role_events = [event for event in events if event["role"] == target_role]
    requested_turn_numbers = extract_requested_turn_numbers(query, total_turns=len(role_events))

    requested_events: list[dict] = []
    seen_turns: set[int] = set()
    for requested_turn in requested_turn_numbers:
        if requested_turn in seen_turns:
            continue
        for event in role_events:
            if event["role_turn"] == requested_turn:
                requested_events.append(event)
                seen_turns.add(requested_turn)
                break

    return target_role, role_events, requested_events


def score_event(query_tokens: list[str], event_content: str) -> float:
    if not query_tokens:
        return 0.0

    event_tokens = tokenize(event_content)
    if not event_tokens:
        return 0.0

    counter = Counter(event_tokens)
    overlap = sum(counter[token] for token in set(query_tokens) if token in counter)
    density = overlap / max(len(event_tokens), 1)
    return overlap + density * 10


def normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def extract_confirmation_candidate(query: str) -> str:
    match = CONFIRMATION_CANDIDATE_PATTERN.search(query)
    if not match:
        return ""

    candidate = match.group("candidate").strip("「」\"'：:，,。.！？? ")
    for prefix in CONFIRMATION_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):].strip("「」\"'：:，,。.！？? ")
            break
    return candidate


def find_best_matching_event(candidate: str, events: list[dict]) -> dict | None:
    normalized_candidate = normalize_match_text(candidate)
    if not normalized_candidate:
        return None

    for event in events:
        normalized_content = normalize_match_text(event["content"])
        if normalized_candidate in normalized_content or normalized_content in normalized_candidate:
            return event

    candidate_tokens = tokenize(candidate)
    if not candidate_tokens:
        return None

    best_match = None
    best_score = 0.0
    for event in events:
        score = score_event(candidate_tokens, event["content"])
        if score > best_score:
            best_match = event
            best_score = score
    return best_match if best_score > 0 else None


def format_event(event: dict) -> str:
    role_label = "使用者" if event["role"] == "user" else "助理"
    attachment_names = [attachment.get("name", "image") for attachment in event.get("attachments", []) if attachment.get("type") == "image"]
    attachment_suffix = f" [附加圖片: {', '.join(attachment_names)}]" if attachment_names else ""
    return f"[{role_label}第{event['role_turn']}則 / event {event['event_index']}{attachment_suffix}]\n{event['content']}"


def format_event_brief(event: dict) -> str:
    attachment_names = [attachment.get("name", "image") for attachment in event.get("attachments", []) if attachment.get("type") == "image"]
    attachment_suffix = f"（附加圖片: {', '.join(attachment_names)}）" if attachment_names else ""
    return f"「{event['content']}」{attachment_suffix}"


def describe_turn(role: str, turn_number: int) -> str:
    if role == "assistant":
        return f"第{turn_number}次回答"
    return f"第{turn_number}個 prompt"


def describe_owner(role: str) -> str:
    return "我" if role == "assistant" else "你的"


def build_precise_history_answer(query: str, messages: list[dict]) -> str | None:
    if not messages or not is_history_query(query):
        return None

    target_role, role_events, requested_events = collect_requested_events(query, messages)
    if not role_events or not requested_events:
        return None

    requested_turns = [event["role_turn"] for event in requested_events]
    if len(requested_turns) == 1:
        target_event = requested_events[0]
        turn_label = f"{describe_owner(target_role)}{describe_turn(target_role, target_event['role_turn'])}"
        confirmation_candidate = extract_confirmation_candidate(query)
        if confirmation_candidate:
            matched_event = find_best_matching_event(confirmation_candidate, role_events)
            if matched_event and matched_event["role_turn"] == target_event["role_turn"]:
                return f"是的，在目前這個 chat 中，{turn_label} 是：\n\n{format_event_brief(target_event)}"
            if matched_event:
                matched_label = f"{describe_owner(target_role)}{describe_turn(target_role, matched_event['role_turn'])}"
                return (
                    f"不是，在目前這個 chat 中，{turn_label} 是：\n\n{format_event_brief(target_event)}\n\n"
                    f"你提到的「{confirmation_candidate}」比較接近{matched_label}：\n\n{format_event_brief(matched_event)}"
                )
            return f"不是，在目前這個 chat 中，{turn_label} 是：\n\n{format_event_brief(target_event)}"

        return f"在目前這個 chat 中，{turn_label} 是：\n\n{format_event_brief(target_event)}"

    lines = ["根據目前這個 chat 的對話紀錄："]
    for event in requested_events:
        lines.append(f"{describe_turn(target_role, event['role_turn'])}：{format_event_brief(event)}")
    return "\n".join(lines)


def build_episodic_memory_context(query: str, messages: list[dict], top_k: int = EPISODIC_MEMORY_TOP_K) -> str:
    if not messages or not is_history_query(query):
        return ""

    events = build_message_events(messages)
    if not events:
        return ""

    _, role_events, requested_events = collect_requested_events(query, messages)
    selected_events: list[dict] = []
    seen_keys: set[tuple[str, int]] = set()

    for event in requested_events:
        key = (event["role"], event["role_turn"])
        if key not in seen_keys:
            selected_events.append(event)
            seen_keys.add(key)

    query_tokens = tokenize(query)
    lexical_candidates = []
    for event in events:
        key = (event["role"], event["role_turn"])
        if key in seen_keys:
            continue
        score = score_event(query_tokens, event["content"])
        if score <= 0:
            continue
        lexical_candidates.append((score, event))

    lexical_candidates.sort(key=lambda item: item[0], reverse=True)
    for _, event in lexical_candidates:
        key = (event["role"], event["role_turn"])
        if key in seen_keys:
            continue
        selected_events.append(event)
        seen_keys.add(key)
        if len(selected_events) >= top_k:
            break

    if not selected_events:
        return ""

    return "【對話事件檢索】\n" + "\n\n".join(format_event(event) for event in selected_events[:top_k])
