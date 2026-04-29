from pathlib import Path


APP_TITLE = "HW2 My Very Powerful Chatbot"
PAGE_LAYOUT = "wide"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
ATTACHMENT_DIR = DATA_DIR / "attachments"
SAVE_FILE = DATA_DIR / "chat_storage.json"
STRUCTURED_MEMORY_FILE = DATA_DIR / "structured_memory.json"

DEFAULT_FOLDER_NAME = "預設群組"
DEFAULT_FOLDERS = [DEFAULT_FOLDER_NAME]
DEFAULT_CHAT_TITLE = "新對話"
DEFAULT_SYSTEM_PROMPT = "請一律使用繁體中文回答。"
DEFAULT_MAX_OUTPUT_TOKENS = 32_768

GENERAL_MODEL_OPTIONS = ["qwen35-4b", "qwen35-397b"]
MODEL_CONTEXT_WINDOWS = {
    "qwen35-4b": 131_072,
    "qwen35-397b": 1_000_000,
}
DEFAULT_MODEL_CONTEXT_WINDOW = 1_000_000
DOCUMENT_FILE_TYPES = ["pdf", "docx", "txt", "py", "c", "h", "cpp", "json", "md"]
IMAGE_FILE_TYPES = ["png", "jpg", "jpeg", "webp"]
VISION_IMAGE_MAX_DIMENSION = 1600
VISION_IMAGE_JPEG_QUALITY = 82

LATEX_FORMAT_RULE = "【格式規範】數學公式請使用 LaTeX (行內 $，段落 $$)。"

DOCUMENT_CHUNK_SIZE = 1200
DOCUMENT_CHUNK_OVERLAP = 200
DOCUMENT_RETRIEVAL_TOP_K = 4
WEB_SEARCH_MAX_RESULTS = 5

RECENT_HISTORY_MESSAGES = 6
EPISODIC_MEMORY_TOP_K = 4
STRUCTURED_MEMORY_TOP_K = 6

COMPLEXITY_KEYWORDS = (
    "分析",
    "比較",
    "整理",
    "總結",
    "解釋",
    "規劃",
    "步驟",
    "architecture",
    "design",
    "compare",
    "explain",
    "analyze",
)

DOCUMENT_HINT_KEYWORDS = (
    "文件",
    "檔案",
    "講義",
    "pdf",
    "doc",
    "根據",
    "依據",
    "資料",
    "uploaded",
    "document",
    "slide",
    "paper",
)

WEB_HINT_KEYWORDS = (
    "最新",
    "今天",
    "近期",
    "新聞",
    "查詢",
    "搜尋",
    "網站",
    "上網",
    "網路",
    "internet",
    "web",
    "search",
    "news",
    "current",
    "latest",
    "recent",
)

IMAGE_HINT_KEYWORDS = (
    "圖",
    "圖片",
    "照片",
    "影像",
    "畫面",
    "圖中",
    "這張圖",
    "那張圖",
    "image",
    "photo",
    "picture",
    "figure",
    "vision",
)

HISTORY_HINT_KEYWORDS = (
    "之前",
    "前面",
    "先前",
    "剛剛",
    "剛才",
    "最早",
    "一開始",
    "第一個",
    "第二個",
    "第三個",
    "第幾個",
    "第幾次",
    "第幾則",
    "prompt",
    "指令",
    "教你",
    "我說",
    "我問",
    "你說",
    "你回答",
    "還記得",
    "history",
    "earlier",
    "previous",
    "first",
    "second",
    "third",
)

CALCULATOR_HINT_KEYWORDS = (
    "計算",
    "算",
    "加",
    "減",
    "乘",
    "除",
    "百分比",
    "%",
    "折扣",
    "稅",
    "換算",
    "calculate",
    "math",
)
