from .config import LATEX_FORMAT_RULE
from .models import DocumentContext


def build_system_prompt(base_prompt: str, document_context: DocumentContext) -> str:
    full_system_prompt = f"{base_prompt}\n\n{LATEX_FORMAT_RULE}"

    if document_context.current_chunk:
        full_system_prompt += (
            f"\n\n【目前指定的資料片段內容】"
            f"\n使用者目前選取並閱讀的是檔案《{document_context.file_name}》的「第 {document_context.chunk_idx_display} 個語意切片」。"
            f"\n請忽略內容中的章節編號與此序號的差異。當使用者說「這段」或「這部分」時，指的就是下方這段內容："
            f"\n[片段開始]\n{document_context.current_chunk}\n[片段結束]"
        )

    return full_system_prompt


def build_messages(chat_messages: list[dict], system_prompt: str) -> list[dict]:
    history = [{"role": message["role"], "content": message["content"]} for message in chat_messages]
    return [{"role": "system", "content": system_prompt}, *history]
