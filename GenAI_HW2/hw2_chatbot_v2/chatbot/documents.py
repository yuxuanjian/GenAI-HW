import hashlib
import io
import uuid
from datetime import datetime
from pathlib import Path

import docx
import fitz
import streamlit as st

from .config import DOCUMENT_CHUNK_OVERLAP, DOCUMENT_CHUNK_SIZE, UPLOAD_DIR
from .storage import save_data


def fingerprint_bytes(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def extract_text_from_bytes(file_bytes: bytes, file_name: str) -> str:
    file_ext = Path(file_name).suffix.lower()

    if file_ext == ".pdf":
        document = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            return "".join(page.get_text() for page in document)
        finally:
            document.close()

    if file_ext == ".docx":
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    return file_bytes.decode("utf-8", errors="ignore")


def chunk_text(raw_text: str, chunk_size: int = DOCUMENT_CHUNK_SIZE, overlap: int = DOCUMENT_CHUNK_OVERLAP) -> list[str]:
    if not raw_text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(raw_text)

    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            last_newline = raw_text.rfind("\n", start, end)
            if last_newline != -1 and last_newline > start:
                end = last_newline + 1

        chunks.append(raw_text[start:end].strip())

        next_start = end - overlap
        start = next_start if next_start < end else end

    return [chunk for chunk in chunks if chunk]


def get_chat_document_library(chat_record: dict) -> dict:
    return chat_record.setdefault("document_library", {})


def ingest_uploaded_documents(chat_id: str, chat_record: dict, uploaded_files: list) -> dict:
    library = get_chat_document_library(chat_record)
    existing_fingerprints = {document.get("fingerprint") for document in library.values()}
    imported: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    chat_upload_dir = UPLOAD_DIR / chat_id
    chat_upload_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        fingerprint = fingerprint_bytes(file_bytes)

        if fingerprint in existing_fingerprints:
            skipped.append(f"{uploaded_file.name} 已存在，跳過重複匯入。")
            continue

        try:
            raw_text = extract_text_from_bytes(file_bytes, uploaded_file.name)
            chunks = chunk_text(raw_text)
            if not chunks:
                errors.append(f"{uploaded_file.name} 沒有可用文字內容。")
                continue

            file_ext = Path(uploaded_file.name).suffix.lower()
            saved_path = chat_upload_dir / f"{fingerprint}{file_ext}"
            if not saved_path.exists():
                saved_path.write_bytes(file_bytes)

            doc_id = str(uuid.uuid4())
            library[doc_id] = {
                "name": uploaded_file.name,
                "file_type": file_ext.lstrip("."),
                "fingerprint": fingerprint,
                "saved_path": str(saved_path),
                "uploaded_at": datetime.now().isoformat(timespec="seconds"),
                "chunk_count": len(chunks),
                "chunks": [{"chunk_index": index, "content": chunk} for index, chunk in enumerate(chunks)],
            }
            existing_fingerprints.add(fingerprint)
            imported.append(uploaded_file.name)
        except Exception as error:
            errors.append(f"{uploaded_file.name} 匯入失敗: {error}")

    if imported:
        save_data()

    return {"imported": imported, "skipped": skipped, "errors": errors}


def delete_document(chat_record: dict, doc_id: str) -> None:
    library = get_chat_document_library(chat_record)
    document = library.get(doc_id)
    if not document:
        return

    saved_path = Path(document.get("saved_path", ""))
    if saved_path.exists():
        try:
            saved_path.unlink()
        except OSError:
            pass

    del library[doc_id]
    save_data()
