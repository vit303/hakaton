"""HTTP ML-сервис по контракту Корсика (docs/ml-contract.md в соседнем репозитории).

Запуск из корня hakaton в окружении torch-cuda:
  python -u scripts/ml_service.py

Бот (hahaton) подключается так:
  KORSIK_ML_MODE=http
  KORSIK_ML_BASE_URL=http://127.0.0.1:9000
  KORSIK_ML_TOKEN=local-rag-token
  KORSIK_ML_TIMEOUT_SECONDS=120
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request  # noqa: E402

from rag.engine import (  # noqa: E402
    NO_INFO,
    analyze_tone,
    ask,
    cluster_trainee_answers,
    knowledge_base,
    vision,
)

app = Flask(__name__)
try:
    app.json.ensure_ascii = False
except Exception:
    app.config["JSON_AS_ASCII"] = False

_lock = threading.Lock()
_ready = False
HOST = os.getenv("ML_HOST", "127.0.0.1")
PORT = int(os.getenv("ML_PORT", "9000"))
TOKEN = os.getenv("ML_TOKEN", os.getenv("KORSIK_ML_TOKEN", "local-rag-token")).strip()
PROCESSOR = "hakaton-rag/1"


def _auth_ok() -> bool:
    if not TOKEN:
        return True
    header = request.headers.get("Authorization") or ""
    if header == f"Bearer {TOKEN}":
        return True
    return False


def _ensure() -> None:
    global _ready
    if _ready:
        return
    vision()
    knowledge_base().embedder.encode(["проверка"], is_query=True)
    _ready = True


def _topics(answer) -> list[str]:
    topics: list[str] = []
    if answer.needs_manager:
        topics.append("manager_support")
    if answer.needs_hr:
        topics.append("hr")
    if answer.mode == "rag":
        topics.append("documents")
    if answer.tone:
        topics.append(answer.tone)
    return topics[:20]


def _map_sources(answer, allowed: list[dict]) -> list[dict]:
    if not allowed or not answer.sources:
        return []
    mapped: list[dict] = []
    seen: set[str] = set()
    for item in answer.sources:
        name = str(item.get("name") or "").lower()
        page = item.get("page")
        page_value = int(page) if isinstance(page, int) and page > 0 else None
        for doc in allowed:
            title = str(doc.get("title") or "").lower()
            doc_id = str(doc.get("id") or "")
            if not doc_id or doc_id in seen:
                continue
            if name and (name in title or title in name or _overlap(name, title)):
                mapped.append(
                    {
                        "document_id": doc_id,
                        "version": int(doc.get("version") or 1),
                        "page": page_value,
                    }
                )
                seen.add(doc_id)
                break
        if len(mapped) >= 5:
            break
    return mapped


def _overlap(left: str, right: str) -> bool:
    left_parts = {part for part in left.replace(".", " ").split() if len(part) >= 4}
    right_parts = {part for part in right.replace(".", " ").split() if len(part) >= 4}
    return bool(left_parts & right_parts)


def _to_answer_payload(answer, allowed: list[dict]) -> dict:
    topics = _topics(answer)
    sources = _map_sources(answer, allowed)
    text = (answer.text or "").strip() or NO_INFO

    if answer.needs_manager:
        note = "Нужно перенаправить к руководителю."
        if note not in text:
            text = text.rstrip() + "\n\n" + note
        return {
            "status": "clarification",
            "answer": text[:12000],
            "sources": [],
            "topics": topics or ["manager_support"],
            "needs_hr": False,
            "processor_version": PROCESSOR,
        }

    if answer.needs_hr:
        return {
            "status": "not_found",
            "answer": text[:12000],
            "sources": [],
            "topics": topics or ["hr"],
            "needs_hr": True,
            "processor_version": PROCESSOR,
        }

    if answer.mode == "rag" and text != NO_INFO:
        if sources:
            return {
                "status": "answered",
                "answer": text[:12000],
                "sources": sources,
                "topics": topics or ["documents"],
                "needs_hr": False,
                "processor_version": PROCESSOR,
            }
        # Ответ из локальной базы есть, но документ ещё не заведён в Корсике —
        # clarification проходит проверку источников и не ломает бота.
        return {
            "status": "clarification",
            "answer": text[:12000],
            "sources": [],
            "topics": topics or ["documents"],
            "needs_hr": False,
            "processor_version": PROCESSOR,
        }

    if answer.mode == "chat":
        return {
            "status": "clarification",
            "answer": text[:12000],
            "sources": [],
            "topics": topics or ["chat"],
            "needs_hr": False,
            "processor_version": PROCESSOR,
        }

    return {
        "status": "not_found",
        "answer": "В базе знаний нет информации по этому вопросу.",
        "sources": [],
        "topics": topics,
        "needs_hr": False,
        "processor_version": PROCESSOR,
    }


@app.get("/health")
def health():
    return jsonify({"ready": _ready, "processor_version": PROCESSOR})


@app.post("/v1/answer")
def v1_answer():
    if not _auth_ok():
        return jsonify({"error": "unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question required"}), 400
    allowed = payload.get("allowed_documents") or []
    if not isinstance(allowed, list):
        allowed = []
    try:
        with _lock:
            _ensure()
            answer = ask(question)
        return jsonify(_to_answer_payload(answer, allowed))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 503


@app.post("/v1/analyze")
def v1_analyze():
    if not _auth_ok():
        return jsonify({"error": "unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text") or "").strip()
    question = str(payload.get("question") or "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400
    try:
        with _lock:
            _ensure()
            tone = analyze_tone(text)
            clusters = cluster_trainee_answers(
                [{"question": question or "Ответ сотрудника", "answer": text}]
            )
        topics = [item["id"] for item in clusters if item["id"] != "ok"][:20]
        if tone.needs_manager and "manager_support" not in topics:
            topics.append("manager_support")
        if tone.hr_topic and "hr_admin" not in topics:
            topics.append("hr_admin")
        if tone.label == "тяжёлый":
            sentiment = "negative"
            risk = "high" if tone.urgent else "medium"
        elif tone.needs_manager or any(
            item in topics for item in ("mentor_conflict", "team_conflict", "overload")
        ):
            sentiment = "negative"
            risk = "medium"
        elif tone.label == "кадры":
            sentiment = "neutral"
            risk = "low"
        else:
            sentiment = "neutral"
            risk = "none"
        explanation = tone.summary or "Разбор тона локальной модели."
        if clusters:
            explanation += " Кластеры: " + ", ".join(item["label"] for item in clusters) + "."
        return jsonify(
            {
                "sentiment": sentiment,
                "risk": risk,
                "topics": topics[:20],
                "explanation": explanation[:3000],
                "needs_hr": bool(tone.needs_hr and not tone.needs_manager),
                "processor_version": PROCESSOR,
            }
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 503


@app.post("/v1/documents")
def v1_documents():
    if not _auth_ok():
        return jsonify({"error": "unauthorized"}), 401
    meta_raw = request.form.get("metadata") or "{}"
    try:
        meta = json.loads(meta_raw)
    except json.JSONDecodeError:
        meta = {}
    document_id = str(meta.get("document_id") or "doc")
    version = int(meta.get("version") or 1)
    external_id = f"{document_id}:{version}"
    # Индексация Корсика остаётся на стороне бота; локальный RAG уже наполнен add_files.
    return jsonify(
        {
            "status": "ready",
            "external_id": external_id,
            "processor_version": PROCESSOR,
        }
    )


@app.get("/v1/documents/<path:external_id>")
def v1_document_status(external_id: str):
    if not _auth_ok():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify(
        {
            "status": "ready",
            "external_id": external_id,
            "processor_version": PROCESSOR,
        }
    )


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    print(f"ML bridge: http://{HOST}:{PORT}", flush=True)
    print(f"Token: {TOKEN or '(any)'}", flush=True)
    with _lock:
        _ensure()
    print(f"Модель готова: {PROCESSOR}", flush=True)
    app.run(host=HOST, port=PORT, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
