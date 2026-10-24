"""Локальная страница, чтобы задавать вопросы базе знаний."""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, abort, jsonify, request, send_file  # noqa: E402

from rag.config import IMAGE_DIR, QWEN_VL_MODEL  # noqa: E402
from rag.engine import (  # noqa: E402
    analyze_tone,
    ask,
    knowledge_base,
    list_files,
    trainee_analysis,
    trainee_questions,
    vision,
)

app = Flask(__name__)
try:
    app.json.ensure_ascii = False
except Exception:
    app.config["JSON_AS_ASCII"] = False

_lock = threading.Lock()
_state = {"ready": False, "error": "", "embedder": False}
HOST = "127.0.0.1"
PORT = 7860


def ensure_ready() -> None:
    if _state["error"]:
        raise RuntimeError(_state["error"])
    vision()
    if not _state["embedder"]:
        knowledge_base().embedder.encode(["проверка"], is_query=True)
        _state["embedder"] = True
    _state["ready"] = True


def _warmup() -> None:
    try:
        with _lock:
            ensure_ready()
        print(f"Модель готова: {QWEN_VL_MODEL}", flush=True)
    except Exception as exc:  # noqa: BLE001
        _state["error"] = str(exc)
        print("Не удалось загрузить модель:", exc, flush=True)
        traceback.print_exc()


def readable_name(name: str) -> str:
    """Имена файлов на диске записаны как UTF-8, прочитанный как CP866."""
    if not name:
        return name
    try:
        repaired = name.encode("cp866").decode("utf-8")
    except UnicodeError:
        return name
    return repaired


def _image_url(path: str) -> str:
    resolved = Path(path).resolve()
    root = IMAGE_DIR.resolve()
    if root not in resolved.parents:
        return ""
    return "/image?path=" + resolved.as_posix()


@app.get("/")
def index():
    return PAGE


@app.get("/api/status")
def status():
    return jsonify(
        {
            "ready": _state["ready"],
            "error": _state["error"],
            "model": QWEN_VL_MODEL,
        }
    )


@app.get("/api/files")
def files():
    rows = list_files()
    for row in rows:
        row["name"] = readable_name(row.get("name") or "")
    return jsonify(rows)


@app.post("/api/ask")
def api_ask():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Введите вопрос."}), 400
    try:
        with _lock:
            ensure_ready()
            answer = ask(question)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    body = answer.to_dict()
    for image in body["images"]:
        image["url"] = _image_url(image.get("path") or "")
        image["source"] = readable_name(image.get("source") or "")
    for source in body["sources"]:
        source["name"] = readable_name(source.get("name") or "")
    return jsonify(body)


@app.post("/api/tone")
def api_tone():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message") or payload.get("question") or "").strip()
    if not message:
        return jsonify({"error": "Введите сообщение."}), 400
    try:
        with _lock:
            ensure_ready()
            tone = analyze_tone(message)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    return jsonify(tone.to_dict())


@app.post("/api/trainee/questions")
def api_trainee_questions():
    try:
        with _lock:
            ensure_ready()
            questions = trainee_questions()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    return jsonify({"questions": questions})


@app.post("/api/trainee/analyze")
def api_trainee_analyze():
    payload = request.get_json(silent=True) or {}
    raw_items = payload.get("items") or []
    if not isinstance(raw_items, list) or not raw_items:
        return jsonify({"error": "Сначала нужны вопросы и ответы."}), 400
    items = []
    for item in raw_items[:3]:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if question:
            items.append({"question": question, "answer": answer})
    if not items:
        return jsonify({"error": "Сначала нужны вопросы и ответы."}), 400
    if not any(item["answer"] for item in items):
        return jsonify({"error": "Напишите ответ хотя бы на один вопрос."}), 400
    try:
        with _lock:
            ensure_ready()
            report = trainee_analysis(items)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    return jsonify(report)


@app.get("/image")
def image():
    raw = request.args.get("path") or ""
    path = Path(raw).resolve()
    root = IMAGE_DIR.resolve()
    if root not in path.parents or not path.is_file():
        abort(404)
    return send_file(path)


PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>База знаний</title>
<style>
  :root { color-scheme: light; --ink: #1c1917; --muted: #57534e; --line: #e7e5e4; --paper: #fafaf9; --accent: #9a3412; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 16px/1.45 "Segoe UI", sans-serif; color: var(--ink); background: var(--paper); }
  header { padding: 20px 28px 8px; }
  h1 { margin: 0 0 4px; font-size: 22px; font-weight: 650; }
  header p { margin: 0; color: var(--muted); }
  main { display: grid; grid-template-columns: 280px 1fr; gap: 20px; padding: 16px 28px 40px; }
  aside, section { background: white; border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
  h2 { margin: 0 0 10px; font-size: 14px; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
  ul { margin: 0; padding: 0; list-style: none; }
  li { padding: 8px 0; border-top: 1px solid var(--line); }
  li:first-child { border-top: 0; }
  .meta { color: var(--muted); font-size: 13px; }
  label { display: block; margin: 12px 0 4px; font-size: 14px; }
  textarea, input { width: 100%; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; font: inherit; }
  textarea { min-height: 96px; resize: vertical; }
  .actions { display: flex; flex-wrap: wrap; gap: 10px; }
  button { margin-top: 14px; background: var(--accent); color: white; border: 0; border-radius: 8px; padding: 10px 16px; font: inherit; cursor: pointer; }
  button.secondary { background: white; color: var(--accent); border: 1px solid var(--accent); }
  button:disabled { opacity: .55; cursor: wait; }
  .trainee { margin-top: 18px; padding-top: 8px; border-top: 1px solid var(--line); }
  .trainee textarea { min-height: 72px; }
  #trainee-result { white-space: pre-wrap; margin-top: 12px; }
  #status { margin-top: 10px; color: var(--muted); }
  #error { color: #b91c1c; white-space: pre-wrap; }
  #answer { white-space: pre-wrap; margin-top: 8px; }
  .hr-note { display: inline-block; margin: 8px 0 0; padding: 6px 10px; border-radius: 999px; background: #fff7ed; color: #9a3412; border: 1px solid #fdba74; font-size: 14px; font-weight: 650; }
  .hr-note[hidden] { display: none; }
  .clusters { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px; }
  .clusters[hidden] { display: none; }
  .cluster { display: inline-block; padding: 5px 10px; border-radius: 999px; background: #f5f5f4; color: #44403c; border: 1px solid var(--line); font-size: 13px; }
  .gallery { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; }
  figure { margin: 0; max-width: 320px; }
  img { width: 100%; border-radius: 8px; border: 1px solid var(--line); background: #fff; }
  figcaption { font-size: 13px; color: var(--muted); margin-top: 4px; }
  @media (max-width: 800px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>База знаний</h1>
  <p id="subtitle">Загрузка модели…</p>
</header>
<main>
  <aside>
    <h2>Файлы</h2>
    <ul id="files"><li class="meta">Загрузка…</li></ul>
  </aside>
  <section>
    <form id="form">
      <label for="question">Вопрос</label>
      <textarea id="question" placeholder="Например: какие цели в политике качества?">Какие обязанности описаны в антикоррупционной политике?</textarea>
      <div class="actions">
        <button id="send" type="submit" disabled>Спросить</button>
        <button id="trainee-start" type="button" class="secondary" disabled>Опрос стажёра</button>
      </div>
      <div id="status"></div>
    </form>
    <div id="trainee" class="trainee" hidden>
      <h2>Стажёр</h2>
      <p class="meta">Три вопроса про самочувствие и адаптацию. После ответов можно разобрать состояние.</p>
      <div id="trainee-questions"></div>
      <button id="trainee-analyze" type="button">Разобрать состояние</button>
      <div id="trainee-manager-note" class="hr-note" hidden>Нужно перенаправить к руководителю</div>
      <div id="trainee-hr-note" class="hr-note" hidden>Нужно перенаправить к HR</div>
      <div id="trainee-clusters" class="clusters" hidden></div>
      <div id="trainee-result"></div>
    </div>
    <div id="error"></div>
    <h2 style="margin-top:22px">Ответ</h2>
    <div id="manager-note" class="hr-note" hidden>Нужно перенаправить к руководителю</div>
    <div id="hr-note" class="hr-note" hidden>Нужно перенаправить к HR</div>
    <div id="answer" class="meta">Ответ появится здесь.</div>
    <div id="gallery" class="gallery"></div>
    <h2 style="margin-top:18px">Источники</h2>
    <ul id="sources"></ul>
  </section>
</main>
<script>
const statusEl = document.querySelector("#status");
const send = document.querySelector("#send");
const traineeStart = document.querySelector("#trainee-start");
const subtitle = document.querySelector("#subtitle");
let modelReady = false;
let busy = false;
function setBusy(value) {
  busy = value;
  send.disabled = value || !modelReady;
  traineeStart.disabled = value || !modelReady;
  const analyze = document.querySelector("#trainee-analyze");
  if (analyze) analyze.disabled = value;
}
async function refreshStatus() {
  const data = await (await fetch("/api/status")).json();
  if (data.error) {
    subtitle.textContent = data.model;
    statusEl.textContent = data.error;
    return;
  }
  subtitle.textContent = data.model;
  modelReady = !!data.ready;
  if (modelReady) {
    if (!busy) setBusy(false);
    if (!statusEl.textContent || statusEl.textContent.indexOf("Загружаю") === 0) {
      statusEl.textContent = "Модель на GPU, можно спрашивать.";
    }
  } else {
    send.disabled = true;
    traineeStart.disabled = true;
    statusEl.textContent = "Загружаю модель на видеокарту. Первый запуск занимает пару минут.";
    setTimeout(refreshStatus, 2000);
  }
}
async function loadFiles() {
  const files = await (await fetch("/api/files")).json();
  const list = document.querySelector("#files");
  list.replaceChildren();
  if (!files.length) {
    list.innerHTML = "<li class='meta'>База пуста. Добавьте файлы скриптом add_files.py</li>";
    return;
  }
  for (const file of files) {
    const item = document.createElement("li");
    item.textContent = file.name;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = "фрагментов: " + file.chunks + ", картинок: " + file.images;
    item.append(meta);
    list.append(item);
  }
}
function fillList(id, rows, render) {
  const list = document.querySelector(id);
  const heading = list.previousElementSibling;
  list.replaceChildren();
  if (!rows.length) {
    list.hidden = true;
    if (heading) heading.hidden = true;
    return;
  }
  list.hidden = false;
  if (heading) heading.hidden = false;
  for (const row of rows) list.append(render(row));
}
document.querySelector("#form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const error = document.querySelector("#error");
  error.textContent = "";
  document.querySelector("#answer").textContent = "Думаю…";
  document.querySelector("#hr-note").hidden = true;
  document.querySelector("#manager-note").hidden = true;
  document.querySelector("#gallery").replaceChildren();
  setBusy(true);
  statusEl.textContent = "Формирую ответ.";
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        question: document.querySelector("#question").value
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Ошибка запроса");
    document.querySelector("#answer").textContent = data.text || "";
    document.querySelector("#hr-note").hidden = !data.needs_hr;
    document.querySelector("#manager-note").hidden = !data.needs_manager;
    const gallery = document.querySelector("#gallery");
    for (const image of data.images || []) {
      if (!image.url) continue;
      const figure = document.createElement("figure");
      const img = document.createElement("img");
      img.src = image.url;
      img.alt = image.source || "Изображение из базы";
      const caption = document.createElement("figcaption");
      caption.textContent = (image.source || "") + (image.page ? ", стр. " + image.page : "");
      figure.append(img, caption);
      gallery.append(figure);
    }
    fillList("#sources", data.sources || [], (source) => {
      const item = document.createElement("li");
      item.textContent = source.name + (source.page ? ", стр. " + source.page : "") + " (" + source.kind + ")";
      return item;
    });
    statusEl.textContent = "Готово. Режим: " + (data.mode || "");
  } catch (err) {
    document.querySelector("#answer").textContent = "";
    document.querySelector("#hr-note").hidden = true;
  document.querySelector("#manager-note").hidden = true;
    error.textContent = err.message;
    statusEl.textContent = "";
  } finally {
    setBusy(false);
  }
});
traineeStart.addEventListener("click", async () => {
  const error = document.querySelector("#error");
  const box = document.querySelector("#trainee");
  const list = document.querySelector("#trainee-questions");
  error.textContent = "";
  document.querySelector("#trainee-result").textContent = "";
  document.querySelector("#trainee-clusters").replaceChildren();
  document.querySelector("#trainee-clusters").hidden = true;
  document.querySelector("#trainee-hr-note").hidden = true;
  document.querySelector("#trainee-manager-note").hidden = true;
  box.hidden = false;
  list.textContent = "Формирую три вопроса…";
  setBusy(true);
  statusEl.textContent = "Составляю вопросы для стажёра.";
  try {
    const response = await fetch("/api/trainee/questions", {method: "POST"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Не удалось составить вопросы");
    list.replaceChildren();
    (data.questions || []).slice(0, 3).forEach((question, index) => {
      const label = document.createElement("label");
      label.textContent = (index + 1) + ". " + question;
      const field = document.createElement("textarea");
      field.dataset.question = question;
      field.placeholder = "Ответ стажёра";
      list.append(label, field);
    });
    statusEl.textContent = "Вопросы готовы. После ответов нажмите «Разобрать состояние».";
  } catch (err) {
    list.replaceChildren();
    box.hidden = true;
    error.textContent = err.message;
    statusEl.textContent = "";
  } finally {
    setBusy(false);
  }
});
document.querySelector("#trainee-analyze").addEventListener("click", async () => {
  const error = document.querySelector("#error");
  const result = document.querySelector("#trainee-result");
  const fields = [...document.querySelectorAll("#trainee-questions textarea")];
  error.textContent = "";
  result.textContent = "Разбираю ответы…";
  setBusy(true);
  statusEl.textContent = "Анализирую состояние и адаптацию.";
  try {
    const response = await fetch("/api/trainee/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        items: fields.map((field) => ({question: field.dataset.question, answer: field.value}))
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Не удалось разобрать ответы");
    result.textContent = data.text || "";
    document.querySelector("#trainee-hr-note").hidden = !data.needs_hr;
    document.querySelector("#trainee-manager-note").hidden = !data.needs_manager;
    const clusterBox = document.querySelector("#trainee-clusters");
    clusterBox.replaceChildren();
    const clusters = data.clusters || [];
    clusterBox.hidden = !clusters.length;
    for (const cluster of clusters) {
      const tag = document.createElement("span");
      tag.className = "cluster";
      tag.textContent = cluster.label || cluster.id || "";
      clusterBox.append(tag);
    }
    statusEl.textContent = "Разбор готов.";
  } catch (err) {
    result.textContent = "";
    document.querySelector("#trainee-clusters").replaceChildren();
    document.querySelector("#trainee-clusters").hidden = true;
    error.textContent = err.message;
    statusEl.textContent = "";
  } finally {
    setBusy(false);
  }
});
loadFiles();
refreshStatus();
</script>
</body>
</html>
"""


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    print(f"Клиент: http://{HOST}:{PORT}", flush=True)
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
