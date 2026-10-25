"""Сопоставление фрагментов RAG с разрешёнными документами Корсика.

Без нечёткого «одно слово совпало»: только точный нормализованный заголовок
или явный korsik_document_id из метаданных индекса.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def repair_mojibake(value: str) -> str:
    """Восстановить UTF-8 имя, ошибочно прочитанное как CP866 (частый сбой на Windows)."""
    text = value or ""
    if not text:
        return text
    # Уже нормальный кириллический текст трогать не нужно.
    if re.search(r"[а-яА-ЯёЁ]", text) and "╨" not in text and "┐" not in text:
        return text
    try:
        return text.encode("cp866").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def normalize_title(value: str) -> str:
    text = repair_mojibake(value or "").strip().lower().replace("ё", "е")
    text = Path(text).stem if "." in Path(text).name and " " not in text[-5:] else text
    text = re.sub(r"[\s_\-–—]+", " ", text)
    text = re.sub(r"[«»\"'()\[\]{}.,;:!?]+", "", text)
    return text.strip()


def titles_match(left: str, right: str) -> bool:
    a = normalize_title(left)
    b = normalize_title(right)
    if not a or not b:
        return False
    return a == b


def resolve_allowed(hit: dict[str, Any] | Any, allowed: list[dict]) -> dict | None:
    """Вернуть элемент allowed_documents, которому принадлежит hit, или None."""
    if not allowed:
        return None
    meta = hit if isinstance(hit, dict) else _hit_as_dict(hit)
    doc_id = str(meta.get("korsik_document_id") or meta.get("document_id") or "").strip()
    version = meta.get("korsik_version")
    name = str(
        meta.get("korsik_title") or meta.get("source_name") or meta.get("name") or ""
    )
    for doc in allowed:
        allowed_id = str(doc.get("id") or "").strip()
        if not allowed_id:
            continue
        if doc_id and doc_id == allowed_id:
            if version is None or int(version) == int(doc.get("version") or 1):
                return doc
            continue
        if not doc_id and titles_match(name, str(doc.get("title") or "")):
            return doc
    return None


def filter_hits(hits: list, allowed: list[dict] | None) -> list:
    if allowed is None:
        return list(hits)
    if not allowed:
        return []
    return [hit for hit in hits if resolve_allowed(hit, allowed) is not None]


def map_sources(hits: list, allowed: list[dict], limit: int = 5) -> list[dict]:
    mapped: list[dict] = []
    seen: set[str] = set()
    for hit in hits:
        doc = resolve_allowed(hit, allowed)
        if not doc:
            continue
        doc_id = str(doc["id"])
        if doc_id in seen:
            continue
        meta = hit if isinstance(hit, dict) else _hit_as_dict(hit)
        page = meta.get("page")
        page_value = int(page) if isinstance(page, int) and page > 0 else None
        mapped.append(
            {
                "document_id": doc_id,
                "version": int(doc.get("version") or 1),
                "page": page_value,
            }
        )
        seen.add(doc_id)
        if len(mapped) >= limit:
            break
    return mapped


def _hit_as_dict(hit: Any) -> dict[str, Any]:
    return {
        "source_name": getattr(hit, "source_name", ""),
        "name": getattr(hit, "source_name", ""),
        "korsik_title": getattr(hit, "korsik_title", ""),
        "page": getattr(hit, "page", 0),
        "korsik_document_id": getattr(hit, "korsik_document_id", ""),
        "korsik_version": getattr(hit, "korsik_version", None),
        "document_id": getattr(hit, "korsik_document_id", ""),
    }
