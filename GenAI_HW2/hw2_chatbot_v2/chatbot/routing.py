import re

from .config import CALCULATOR_HINT_KEYWORDS, COMPLEXITY_KEYWORDS, DOCUMENT_HINT_KEYWORDS, IMAGE_HINT_KEYWORDS, WEB_HINT_KEYWORDS
from .history import is_history_query
from .models import RouteDecision


ARITHMETIC_PATTERN = re.compile(r"[\d\.\s\+\-\*\/\(\)%×÷\^]{5,}")


def is_complex_prompt(prompt: str) -> bool:
    normalized = prompt.lower()
    if len(prompt) > 120 or "\n" in prompt:
        return True
    return any(keyword.lower() in normalized for keyword in COMPLEXITY_KEYWORDS)


def should_use_calculator(prompt: str) -> bool:
    normalized = prompt.lower()
    arithmetic_match = ARITHMETIC_PATTERN.search(prompt)
    if arithmetic_match and re.search(r"[\+\*\/%×÷\^]", arithmetic_match.group(0)):
        return True
    if arithmetic_match and "-" in arithmetic_match.group(0) and any(keyword.lower() in normalized for keyword in CALCULATOR_HINT_KEYWORDS):
        return True
    return any(keyword.lower() in normalized for keyword in CALCULATOR_HINT_KEYWORDS)


def should_use_document_retrieval(prompt: str, document_names: list[str]) -> bool:
    if not document_names:
        return False

    normalized = prompt.lower()
    if any(keyword.lower() in normalized for keyword in DOCUMENT_HINT_KEYWORDS):
        return True

    for name in document_names:
        file_stem = name.rsplit(".", maxsplit=1)[0].lower()
        if len(file_stem) >= 3 and file_stem in normalized:
            return True

    return False


def should_use_web_search(prompt: str) -> bool:
    normalized = prompt.lower()
    return any(keyword.lower() in normalized for keyword in WEB_HINT_KEYWORDS)


def should_use_visual_context(prompt: str) -> bool:
    normalized = prompt.lower()
    return any(keyword.lower() in normalized for keyword in IMAGE_HINT_KEYWORDS)


def decide_route(
    prompt: str,
    has_new_image: bool,
    has_image_context: bool,
    web_search_enabled: bool,
    document_names: list[str],
    small_model: str,
    large_model: str,
    vision_model: str,
) -> RouteDecision:
    tools: list[str] = []
    reasons: list[str] = []
    selected_model = ""
    use_vision = False

    if has_new_image:
        selected_model = vision_model or large_model
        use_vision = True
        reasons.append("偵測到圖片附件，優先使用 vision model。")
    elif has_image_context and should_use_visual_context(prompt):
        selected_model = vision_model or large_model
        use_vision = True
        reasons.append("目前對話已有圖片上下文，且本輪問題指向圖片，因此沿用 vision model。")

    if should_use_calculator(prompt):
        tools.append("calculator")
        reasons.append("輸入包含計算特徵，啟用 calculator。")

    if should_use_document_retrieval(prompt, document_names):
        tools.append("document_retrieval")
        reasons.append("輸入看起來依賴上傳文件，啟用 document retrieval。")

    if web_search_enabled:
        tools.append("web_search")
        reasons.append("手動開啟 web search，這一輪使用網路搜尋。")

    if is_history_query(prompt):
        tools.append("history_lookup")
        reasons.append("輸入在詢問先前對話內容或順序，啟用 history lookup。")

    if not selected_model:
        if tools or is_complex_prompt(prompt):
            selected_model = large_model
            reasons.append("問題較複雜或需要工具，因此使用大模型。")
        else:
            selected_model = small_model
            reasons.append("一般短問答，使用小模型以降低成本。")

    unique_tools = list(dict.fromkeys(tools))
    return RouteDecision(
        selected_model=selected_model,
        tools=unique_tools,
        use_vision=use_vision,
        reason=" ".join(reasons),
    )
