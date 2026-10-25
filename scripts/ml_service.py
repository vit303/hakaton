"""HTTP ML-сервис по контракту Корсика (docs/ml-contract.md в соседнем репозитории).

Запуск из корня hakaton в окружении torch-cuda:
  python -u scripts/ml_service.py

По умолчанию слушает 0.0.0.0:9000 — доступно с хоста и из Docker
через host.docker.internal / IP хоста.

Бот (hahaton):
  KORSIK_ML_MODE=http
  KORSIK_ML_BASE_URL=http://127.0.0.1:9000   # или http://host.docker.internal:9000
  KORSIK_ML_TOKEN=local-rag-token
  KORSIK_ML_TIMEOUT_SECONDS=120
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_file  # noqa: E402

from rag.access import map_sources  # noqa: E402
from rag.engine import (  # noqa: E402
    NO_INFO,
    analyze_tone,
    ask,
    cluster_trainee_answers,
    index_uploaded_file,
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
# 0.0.0.0 — чтобы Docker/другие машины видели сервис; для чисто локального режима задайте ML_HOST=127.0.0.1
HOST = os.getenv("ML_HOST", "0.0.0.0")
PORT = int(os.getenv("ML_PORT", "9000"))
TOKEN = os.getenv("ML_TOKEN", os.getenv("KORSIK_ML_TOKEN", "local-rag-token")).strip()
PROCESSOR = "hakaton-rag/2"
UPLOAD_DIR = ROOT / "data" / "korsik_uploads"
MEDIA_DIR = ROOT / "data" / "korsik_media"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

_index_jobs: dict[str, dict] = {}
_media: dict[str, Path] = {}
_jobs_lock = threading.Lock()
_JOBS_PATH = UPLOAD_DIR / "index_jobs.json"


def _load_jobs() -> None:
    if not _JOBS_PATH.exists():
        return
    try:
        data = json.loads(_JOBS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _index_jobs.update(data)
    except Exception:  # noqa: BLE001
        pass


def _save_jobs() -> None:
    _JOBS_PATH.write_text(json.dumps(_index_jobs, ensure_ascii=False, indent=2), encoding="utf-8")


_load_jobs()


def _auth_ok() -> bool:
    if not TOKEN:
        return True
    header = request.headers.get("Authorization") or ""
    return header == f"Bearer {TOKEN}"


def _ensure() -> None:
    global _ready
    if _ready:
        return
    vision()
    knowledge_base().embedder.encode(["проверка"], is_query=True)
    _ready = True


def _public_base() -> str:
    configured = (os.getenv("ML_PUBLIC_BASE_URL") or "").rstrip("/")
    if configured:
        return configured
    # Для ссылок на медиа в ответе; из Docker бот ходит на host.docker.internal.
    host = os.getenv("ML_PUBLIC_HOST", "127.0.0.1")
    return f"http://{host}:{PORT}"


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


def _register_images(answer) -> list[dict]:
    """Копирует картинки ответа в доступное по HTTP хранилище."""
    refs: list[dict] = []
    for image in answer.images or []:
        path = Path(str(image.get("path") or ""))
        if not path.is_file():
            continue
        media_id = secrets.token_urlsafe(12)
        target = MEDIA_DIR / f"{media_id}{path.suffix.lower() or '.jpg'}"
        target.write_bytes(path.read_bytes())
        _media[media_id] = target
        refs.append(
            {
                "media_id": media_id,
                "url": f"{_public_base()}/v1/media/{media_id}",
                "page": image.get("page") if isinstance(image.get("page"), int) else None,
                "document_id": None,
            }
        )
        if len(refs) >= 3:
            break
    # Привязка document_id по странице/имени источника, если источники уже смаплены.
    return refs


def _bind_image_docs(images: list[dict], sources: list[dict], answer) -> list[dict]:
    if not images or not sources:
        return images
    by_page = {s.get("page"): s.get("document_id") for s in sources if s.get("page")}
    for image, raw in zip(images, answer.images or []):
        page = raw.get("page")
        if page in by_page:
            image["document_id"] = by_page[page]
        elif sources:
            image["document_id"] = sources[0].get("document_id")
    return images


def _to_answer_payload(answer, allowed: list[dict]) -> dict:
    topics = _topics(answer)
    text = (answer.text or "").strip() or NO_INFO

    if answer.needs_manager:
        return {
            "status": "clarification",
            "answer": text[:12000],
            "sources": [],
            "topics": topics or ["manager_support"],
            "needs_hr": False,
            "images": [],
            "processor_version": PROCESSOR,
        }

    if answer.needs_hr and answer.mode == "hr":
        return {
            "status": "not_found",
            "answer": text[:12000],
            "sources": [],
            "topics": topics or ["hr"],
            "needs_hr": True,
            "images": [],
            "processor_version": PROCESSOR,
        }

    if answer.mode == "rag":
        # Источники только из ACL; без подтверждённого источника — честный not_found.
        if allowed:
            if answer.sources and all("document_id" in item for item in answer.sources):
                sources = [
                    {
                        "document_id": str(item["document_id"]),
                        "version": int(item.get("version") or 1),
                        "page": item.get("page") if isinstance(item.get("page"), int) else None,
                    }
                    for item in answer.sources
                    if any(str(doc.get("id")) == str(item.get("document_id")) for doc in allowed)
                ]
            else:
                sources = map_sources(
                    [
                        {
                            "source_name": item.get("name"),
                            "name": item.get("name"),
                            "page": item.get("page"),
                            "korsik_document_id": item.get("document_id"),
                            "korsik_version": item.get("version"),
                        }
                        for item in (answer.sources or [])
                    ],
                    allowed,
                )
        else:
            sources = []

        if text == NO_INFO or not sources:
            return {
                "status": "not_found",
                "answer": "В доступных документах нет подтверждённого ответа по этому вопросу.",
                "sources": [],
                "topics": topics or ["documents"],
                "needs_hr": True,
                "images": [],
                "processor_version": PROCESSOR,
            }

        images = _bind_image_docs(_register_images(answer), sources, answer)
        return {
            "status": "answered",
            "answer": text[:12000],
            "sources": sources,
            "topics": topics or ["documents"],
            "needs_hr": False,
            "images": images,
            "processor_version": PROCESSOR,
        }

    if answer.mode == "chat":
        return {
            "status": "clarification",
            "answer": text[:12000],
            "sources": [],
            "topics": topics or ["chat"],
            "needs_hr": False,
            "images": [],
            "processor_version": PROCESSOR,
        }

    return {
        "status": "not_found",
        "answer": "В базе знаний нет информации по этому вопросу.",
        "sources": [],
        "topics": topics,
        "needs_hr": True,
        "images": [],
        "processor_version": PROCESSOR,
    }


@app.get("/health")
def health():
    return jsonify({"ready": _ready, "processor_version": PROCESSOR, "host": HOST, "port": PORT})


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
    history = payload.get("history") or []
    if not isinstance(history, list):
        history = []
    access = payload.get("access") if isinstance(payload.get("access"), dict) else {}
    step_context = payload.get("step_context") if isinstance(payload.get("step_context"), dict) else {}
    address_style = str(payload.get("address_style") or "formal")
    try:
        with _lock:
            _ensure()
            answer = ask(
                question,
                allowed_documents=allowed,
                history=history,
                access=access,
                step_context=step_context,
                address_style=address_style,
            )
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
        topics = [item["id"] for item in clusters][:20]
        uncertain = "classification_uncertain" in topics or not clusters
        if uncertain:
            topics = ["classification_uncertain"]
        else:
            topics = [item for item in topics if item != "ok"] or (
                ["ok"] if any(c["id"] == "ok" for c in clusters) else ["classification_uncertain"]
            )
        if tone.needs_manager and "manager_support" not in topics:
            topics.append("manager_support")
        if tone.hr_topic and "hr_admin" not in topics:
            topics.append("hr_admin")
        if uncertain:
            sentiment = "neutral"
            risk = "medium"
            needs_hr = True
            explanation = "Модель не смогла уверенно классифицировать ответ; требуется проверка HR."
        elif tone.label == "тяжёлый":
            sentiment = "negative"
            risk = "high" if tone.urgent else "medium"
            needs_hr = bool(tone.needs_hr and not tone.needs_manager)
            explanation = tone.summary or "Разбор тона локальной модели."
        elif tone.needs_manager or any(
            item in topics for item in ("mentor_conflict", "team_conflict", "overload")
        ):
            sentiment = "negative"
            risk = "medium"
            needs_hr = bool(tone.needs_hr and not tone.needs_manager)
            explanation = tone.summary or "Разбор тона локальной модели."
        elif tone.label == "кадры":
            sentiment = "neutral"
            risk = "low"
            needs_hr = True
            explanation = tone.summary or "Разбор тона локальной модели."
        else:
            sentiment = "neutral"
            risk = "none"
            needs_hr = bool(tone.needs_hr and not tone.needs_manager)
            explanation = tone.summary or "Разбор тона локальной модели."
        if clusters and not uncertain:
            explanation += " Кластеры: " + ", ".join(item["label"] for item in clusters) + "."
        return jsonify(
            {
                "sentiment": sentiment,
                "risk": risk,
                "topics": topics[:20],
                "explanation": explanation[:3000],
                "needs_hr": needs_hr,
                "processor_version": PROCESSOR,
            }
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 503


def _run_index(external_id: str, path: Path, meta: dict) -> None:
    try:
        with _lock:
            _ensure()
            index_uploaded_file(
                path,
                document_id=str(meta.get("document_id") or "doc"),
                version=int(meta.get("version") or 1),
                title=str(meta.get("title") or path.name),
                branch_codes=meta.get("branch_codes") or [],
                categories=meta.get("categories") or [],
                caption=False,
                log=lambda *_args, **_kwargs: None,
            )
        with _jobs_lock:
            job = _index_jobs.get(external_id) or {}
            job.update({"status": "ready", "error": None, "processor_version": PROCESSOR})
            _index_jobs[external_id] = job
            _save_jobs()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        with _jobs_lock:
            job = _index_jobs.get(external_id) or {}
            job.update({"status": "failed", "error": str(exc), "processor_version": PROCESSOR})
            _index_jobs[external_id] = job
            _save_jobs()


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
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "file required", "status": "failed", "external_id": external_id}), 400
    suffix = Path(upload.filename).suffix.lower() or ".bin"
    target = UPLOAD_DIR / f"{document_id}_v{version}{suffix}"
    upload.save(target)
    with _jobs_lock:
        _index_jobs[external_id] = {
            "status": "processing",
            "external_id": external_id,
            "path": str(target),
            "meta": meta,
            "error": None,
            "processor_version": PROCESSOR,
        }
        _save_jobs()
    threading.Thread(target=_run_index, args=(external_id, target, meta), daemon=True).start()
    return jsonify(
        {
            "status": "processing",
            "external_id": external_id,
            "processor_version": PROCESSOR,
        }
    )


@app.get("/v1/documents/<path:external_id>")
def v1_document_status(external_id: str):
    if not _auth_ok():
        return jsonify({"error": "unauthorized"}), 401
    with _jobs_lock:
        job = _index_jobs.get(external_id)
    if not job:
        # После рестарта: если чанки с этим document_id уже в индексе — ready.
        document_id = external_id.split(":", 1)[0]
        present = any(
            str(item.get("metadata", {}).get("korsik_document_id") or "") == document_id
            for item in knowledge_base()._load()
        )
        status = "ready" if present else "failed"
        return jsonify(
            {
                "status": status,
                "external_id": external_id,
                "processor_version": PROCESSOR,
            }
        )
    return jsonify(
        {
            "status": job["status"],
            "external_id": external_id,
            "processor_version": job.get("processor_version") or PROCESSOR,
        }
    )


@app.get("/v1/media/<media_id>")
def v1_media(media_id: str):
    if not _auth_ok():
        return jsonify({"error": "unauthorized"}), 401
    path = _media.get(media_id)
    if not path or not path.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/jpeg", download_name=path.name)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    print(f"ML bridge: http://{HOST}:{PORT}", flush=True)
    print(f"Public media base: {_public_base()}", flush=True)
    print(f"Token: {TOKEN or '(any)'}", flush=True)
    with _lock:
        _ensure()
    print(f"Модель готова: {PROCESSOR}", flush=True)
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
