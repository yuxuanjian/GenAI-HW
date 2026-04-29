from dataclasses import dataclass
from pathlib import Path


APP_TITLE = "My Chatbot"
PAGE_LAYOUT = "wide"
BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_FILE = BASE_DIR / "chat_storage.json"

DEFAULT_FOLDER_NAME = "預設群組"
DEFAULT_FOLDERS = [DEFAULT_FOLDER_NAME]
DEFAULT_CHAT_TITLE = "新對話"
DEFAULT_SYSTEM_PROMPT = "你是一個專業助理。請用繁體中文回答"

MODEL_OPTIONS = ["qwen35-397b", "qwen35-4b"]
UPLOADER_FILE_TYPES = ["pdf", "docx", "txt", "py", "c", "h", "cpp", "json", "md"]
LATEX_FORMAT_RULE = "【格式規範】數學公式請使用 LaTeX (行內 $，段落 $$)。"


@dataclass(frozen=True)
class ChunkPolicy:
    chunk_size: int
    overlap: int


CHUNK_POLICIES = {
    "qwen35-4b": ChunkPolicy(chunk_size=1500, overlap=200),
    "qwen35-397b": ChunkPolicy(chunk_size=6000, overlap=500),
}


def get_chunk_policy(model_name: str) -> ChunkPolicy:
    return CHUNK_POLICIES.get(model_name, CHUNK_POLICIES["qwen35-397b"])
