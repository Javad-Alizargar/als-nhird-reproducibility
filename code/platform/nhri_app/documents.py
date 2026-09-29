from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import SOURCE_FILES


@dataclass
class DocumentChunk:
    source: str
    title: str
    text: str


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - runtime dependency message
        raise RuntimeError("Install pypdf to read PDF files: pip install pypdf") from exc
    reader = PdfReader(str(path))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {idx}]\n{text}")
    return "\n\n".join(pages)


def chunk_text(source: str, title: str, text: str, max_chars: int = 1600, overlap: int = 220) -> list[DocumentChunk]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        window = normalized[start:end]
        chunks.append(DocumentChunk(source=source, title=title, text=window))
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def load_knowledge_documents(source_dir: Path) -> list[DocumentChunk]:
    docs: list[DocumentChunk] = []
    text_sources = {
        "catalog": SOURCE_FILES["catalog"],
        "money": SOURCE_FILES["money"],
        "review": SOURCE_FILES["review"],
        "qa": SOURCE_FILES["qa"],
        "rare_plan": SOURCE_FILES["rare_plan"],
    }
    for label, filename in text_sources.items():
        path = source_dir / filename
        if path.exists():
            docs.extend(chunk_text(filename, label, read_text_file(path)))
    for label in ("datasets_pdf", "regulation_pdf"):
        filename = SOURCE_FILES[label]
        path = source_dir / filename
        if path.exists():
            docs.extend(chunk_text(filename, label, read_pdf(path)))
    return docs


def available_sources(source_dir: Path) -> Iterable[tuple[str, bool]]:
    for label, filename in SOURCE_FILES.items():
        if label == "api_keys":
            continue
        yield filename, (source_dir / filename).exists()
