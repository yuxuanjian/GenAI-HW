import io

import docx
import fitz

from .config import get_chunk_policy
from .models import DocumentContext


def extract_text_from_upload(uploaded_file) -> str:
    file_name = uploaded_file.name
    file_ext = file_name.rsplit(".", maxsplit=1)[-1].lower()
    file_bytes = uploaded_file.getvalue()

    if file_ext == "pdf":
        document = fitz.open(stream=file_bytes, filetype="pdf")
        return "".join(page.get_text() for page in document)

    if file_ext == "docx":
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    return file_bytes.decode("utf-8")


def chunk_text(raw_text: str, model_name: str) -> list[str]:
    if not raw_text:
        return []

    policy = get_chunk_policy(model_name)
    chunks: list[str] = []
    start = 0
    text_len = len(raw_text)

    while start < text_len:
        end = start + policy.chunk_size
        if end < text_len:
            last_newline = raw_text.rfind("\n", start, end)
            if last_newline != -1 and last_newline > start:
                end = last_newline + 1

        chunks.append(raw_text[start:end].strip())

        next_start = end - policy.overlap
        start = next_start if next_start < end else end

    return chunks


def build_document_context(uploaded_file, model_name: str) -> DocumentContext:
    raw_text = extract_text_from_upload(uploaded_file)
    chunks = chunk_text(raw_text, model_name)

    document_context = DocumentContext(
        file_name=uploaded_file.name,
        chunks=chunks,
        total_chunks=len(chunks),
    )

    if chunks:
        document_context.select_chunk(0)

    return document_context
