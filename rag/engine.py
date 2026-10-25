"""Добавление файлов, удаление и ответы по базе знаний."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from rag.access import filter_hits, map_sources, repair_mojibake
from rag.config import (
    HR_CONTACT,
    IMAGE_DIR,
    LAST_ANSWER_JSON,
    LAST_ANSWER_MD,
    MAX_ANSWER_IMAGES,
    MAX_IMAGES_PER_FILE,
    MAX_NEW_TOKENS,
    MIN_IMAGE_SIDE,
    ROOT,
    SKIP_DIRS,
    SUPPORTED_EXTENSIONS,
    TOP_K,
)
from rag.documents import Block, parse_file
from rag.store import (
    Hit,
    KnowledgeBase,
    forget_file,
    load_manifest,
    make_id,
    remember_file,
)
from rag.vl import VisionModel


@dataclass
class StoredChunk:
    text: str
    source: str
    source_name: str
    kind: str
    page: int
    image_path: str = ""


@dataclass
class Answer:
    text: str
    images: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    mode: str = "rag"
    tone: str = ""
    needs_hr: bool = False
    needs_manager: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


_KB: KnowledgeBase | None = None
_VL: VisionModel | None = None


def knowledge_base() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB


_TRAINEE_FALLBACK = (
    "Как вы себя чувствуете в последние дни: настроение, силы и напряжение?",
    "Как идёт адаптация: понятны ли задачи и хватает ли поддержки наставника и команды?",
    "Что сейчас помогает освоиться, а что больше всего мешает?",
)

# Кластеры проблем адаптации. Модель выбирает по смыслу ответов, без заучивания фраз.
TRAINEE_CLUSTERS: tuple[tuple[str, str], ...] = (
    ("access", "Проблема с доступом"),
    ("tools", "Проблемы с инструментами"),
    ("mentor_conflict", "Конфликт с наставником"),
    ("team_conflict", "Конфликт с командой"),
    ("unclear_tasks", "Неясные задачи"),
    ("overload", "Перегрузка и усталость"),
    ("no_support", "Нехватка поддержки"),
    ("isolation", "Одиночество и изоляция"),
    ("learning", "Сложности с обучением"),
    ("ok", "Адаптация идёт нормально"),
)
_CLUSTER_HINTS = {
    "access": "нет доступа, прав, учётки, VPN, почты, пропуска",
    "tools": "ломается ПО, неудобные инструменты, баги",
    "mentor_conflict": "ссора или напряжение с наставником",
    "team_conflict": "ссора или напряжение с командой или коллегами",
    "unclear_tasks": "непонятны задачи, цели, ожидания",
    "overload": "усталость, слишком много работы",
    "no_support": "некому спросить, мало помощи",
    "isolation": "одиноко, не принимают в коллектив",
    "learning": "сложно учиться, много нового сразу",
    "ok": "трудностей нет, адаптация спокойная",
}
_CLUSTER_BY_ID = {key: label for key, label in TRAINEE_CLUSTERS}
_CLUSTER_BY_LABEL = {label.lower(): key for key, label in TRAINEE_CLUSTERS}


def trainee_questions() -> list[str]:
    """Три вопроса стажёру: самочувствие и ход адаптации. База знаний не используется."""
    raw = vision().generate(
        "Составь ровно 3 вопроса стажёру. "
        "Первый — про психологическое состояние: настроение, силы, напряжение. "
        "Второй — как идёт адаптация: задачи, команда, наставник. "
        "Третий — что помогает и что мешает освоиться. "
        "Каждый вопрос с новой строки, с номером и знаком вопроса. Без вступления и пояснений.",
        system=(
            "Ты составляешь короткий доброжелательный опрос для стажёра. "
            "Это не диагностика. Не упоминай базу знаний, документы и источники. Пиши на русском."
        ),
        max_new_tokens=160,
    )
    found: list[str] = []
    for line in raw.splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-•*]\s*)", "", line).strip()
        if not cleaned:
            continue
        pieces = re.findall(r"[^?]{12,}\?", cleaned) or ([cleaned] if len(cleaned) >= 12 else [])
        for piece in pieces:
            question = re.sub(r"\s+", " ", piece).strip()
            if question not in found:
                found.append(question)
            if len(found) == 3:
                break
        if len(found) == 3:
            break
    for question in _TRAINEE_FALLBACK:
        if len(found) >= 3:
            break
        if question not in found:
            found.append(question)
    return found[:3]


def _strip_unearned_calm(text: str, answers: str) -> str:
    signals = ("уста", "роб", "меша", "трудн", "напряж", "тревож", "волн", "одинок")
    if not any(signal in answers.lower() for signal in signals):
        return text
    cleaned = re.sub(r",?\s*(?:нет|без)\s+признаков[^.!\n]*", "", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([.!?])", r"\1", cleaned)
    cleaned = re.sub(r"([.!?]){2,}", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip() or text


def _parse_clusters(raw: str) -> list[dict]:
    chosen: list[dict] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-•*]\s*)", "", line).strip()
        if not cleaned:
            continue
        if cleaned.lower().startswith("кластер"):
            cleaned = cleaned.split(":", 1)[-1].strip()
        for part in re.split(r"[;|,/]+", cleaned):
            piece = part.strip().strip("«»\"'")
            if not piece:
                continue
            key = ""
            label = ""
            lowered = piece.lower()
            if lowered in _CLUSTER_BY_ID:
                key = lowered
                label = _CLUSTER_BY_ID[key]
            elif lowered in _CLUSTER_BY_LABEL:
                key = _CLUSTER_BY_LABEL[lowered]
                label = _CLUSTER_BY_ID[key]
            else:
                for cluster_id, cluster_label in TRAINEE_CLUSTERS:
                    if cluster_label.lower() in lowered or lowered in cluster_label.lower():
                        key = cluster_id
                        label = cluster_label
                        break
            if not key or key in seen:
                continue
            seen.add(key)
            chosen.append({"id": key, "label": label})
    if not chosen:
        # Не подменяем сбой/мусор модели на «всё хорошо».
        return [{"id": "classification_uncertain", "label": "Классификация неуверенна"}]
    if any(item["id"] != "ok" for item in chosen):
        chosen = [item for item in chosen if item["id"] != "ok"]
    return chosen


def cluster_trainee_answers(items: list[dict]) -> list[dict]:
    """Кластеризация ответов стажёра по типам проблем адаптации."""
    lines = []
    for index, item in enumerate(items, start=1):
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip() or "ответ не дан"
        lines.append(f"{index}. {question}\nОтвет: {answer}")
    catalog = "\n".join(
        f"- {key}: {label} ({_CLUSTER_HINTS.get(key, '')})" for key, label in TRAINEE_CLUSTERS
    )
    raw = vision().generate(
        "Ответы стажёра:\n\n"
        + "\n\n".join(lines)
        + "\n\nДоступные кластеры:\n"
        + catalog
        + "\n\nВыбери только те кластеры, которые прямо следуют из ответов. "
        "Если нет доступа или прав — это access, а не tools. "
        "Перечисли их id через точку с запятой в одной строке. Без пояснений.",
        system=(
            "Ты размечаешь ответы стажёра по кластерам проблем адаптации. "
            "Смотри смысл, не заучивай формулировки. Можно выбрать несколько кластеров. "
            "Если трудностей нет, верни только ok. Пиши только id из списка."
        ),
        max_new_tokens=80,
    )
    return _parse_clusters(raw)


def trainee_analysis(items: list[dict]) -> dict:
    """Разбор ответов стажёра и кластеры проблем. Это не медицинское заключение."""
    lines = []
    answers = []
    for index, item in enumerate(items, start=1):
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip() or "ответ не дан"
        answers.append(answer)
        lines.append(f"{index}. {question}\nОтвет: {answer}")
    clusters = cluster_trainee_answers(items)
    cluster_names = ", ".join(item["label"] for item in clusters) or "не определены"
    text = vision().generate(
        "Ответы стажёра:\n\n"
        + "\n\n".join(lines)
        + f"\n\nУже выделенные кластеры: {cluster_names}",
        system=(
            "Ты помогаешь наставнику понять, как стажёр проходит адаптацию. "
            "Пиши на русском, коротко, не больше 120 слов, только по ответам. Это не диагноз и не медицинское заключение. "
            "Структура: психологическое состояние; как идёт адаптация; кластеры проблем одной строкой; "
            "на что наставнику обратить внимание. "
            "Не выдумывай факты, которых нет в ответах. Если стажёр пишет про усталость, робость или трудности, отрази их и не пиши, что стресса нет. Базу знаний и документы не используй. "
            "Если в ответах есть мысль о том, что человеку плохо настолько, что нужна срочная помощь, "
            "мягко напиши обратиться к наставнику и в экстренную службу 112. Способы вреда не описывай."
        ),
        max_new_tokens=260,
    )
    text = _strip_unearned_calm(text, "\n".join(answers))
    tone = analyze_tone("\n".join(answers))
    problem_ids = {item["id"] for item in clusters}
    needs_manager = tone.needs_manager or ("mentor_conflict" in problem_ids) or ("team_conflict" in problem_ids)
    needs_hr = tone.needs_hr and not needs_manager
    return {
        "text": text,
        "clusters": clusters,
        "needs_hr": needs_hr,
        "needs_manager": needs_manager,
        "tone": tone.label,
        "hr_topic": tone.hr_topic,
    }


_URGENT_RE = re.compile(
    r"не\s+хочу\s+жить|суицид|покончить\s+с\s+собой|убить\s+себя|нет\s+смысла\s+жить",
    re.IGNORECASE,
)


HR_NO_INFO = "В базе знаний нет информации, перенаправляю к HR."


@dataclass
class Tone:
    label: str
    needs_hr: bool
    needs_manager: bool
    urgent: bool
    hr_topic: bool
    summary: str
    reply: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tone"] = self.label
        return data


def _hr_distress_reply(urgent: bool) -> str:
    text = "Слышу, что вам сейчас тяжело."
    if urgent:
        text += " Если нужна срочная помощь прямо сейчас, позвоните 112."
    return text


def _flag(raw: str, key: str) -> bool | None:
    match = re.search(rf"{key}\s*:\s*(да|нет)", raw or "", re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "да"


def analyze_tone(message: str) -> Tone:
    """Контекст: коллектив → руководитель; зарплата/увольнение и тяжёлое состояние → HR."""
    message = (message or "").strip()
    if not message:
        raise ValueError("Пустое сообщение.")
    urgent = bool(_URGENT_RE.search(message))
    raw = vision().generate(
        f"Сообщение сотрудника:\n{message}",
        system=(
            "Ты только размечаешь контекст для бота. Не отвечай сотруднику и не запоминай шаблоны фраз. "
            "Смотри смысл. Ответь ровно тремя строками:\n"
            "коллектив: да|нет\n"
            "кадры: да|нет\n"
            "состояние: да|нет\n"
            "коллектив: да — ссора, ругань, конфликт, претензии, проблемы с коллегами, командой или коллективом. "
            "кадры: да — зарплата, оклад, премия, бонус, ставка, ЗП, увольнение, сокращение, отпуск, "
            "больничный, кадровый учёт, перевод, ставка, оформление, выходное пособие, расчёт при уходе. "
            "состояние: да — самому человеку плохо, нет сил, страшно, подавлен, выгорел. "
            "Вопрос по политике компании, приветствие, обычная просьба: все три нет."
        ),
        max_new_tokens=64,
    )
    team = bool(_flag(raw, "коллектив"))
    hr_topic = bool(_flag(raw, "кадры"))
    state = bool(_flag(raw, "состояние"))
    needs_manager = team and not hr_topic
    needs_hr = urgent or hr_topic or (state and not needs_manager)
    if needs_manager:
        label = "конфликт"
        summary = "Проблема с коллегами или коллективом."
        reply = ""
    elif hr_topic:
        label = "кадры"
        summary = "Вопрос про зарплату, увольнение или другие кадровые темы."
        reply = HR_NO_INFO
    elif needs_hr:
        label = "тяжёлый"
        summary = "Сотруднику сейчас тяжело."
        reply = _hr_distress_reply(urgent)
    else:
        label = "спокойный"
        summary = "Обычное сообщение."
        reply = ""
    return Tone(
        label=label,
        needs_hr=needs_hr,
        needs_manager=needs_manager,
        urgent=urgent,
        hr_topic=hr_topic,
        summary=summary,
        reply=reply,
    )


def vision() -> VisionModel:
    global _VL
    if _VL is None:
        _VL = VisionModel()
    return _VL


def add_files(paths: list[str | Path], caption: bool = True, log=print) -> list[dict]:
    files = _collect(paths)
    if not files:
        raise FileNotFoundError("Нет файлов поддерживаемого формата.")
    reports = []
    for path in files:
        log(f"Добавляю {path.name}")
        report = _add_one(path, caption=caption, log=log)
        reports.append(report)
        log(f"  фрагментов: {report['chunks']}, картинок: {report['images']}")
    return reports


def remove_files(paths: list[str | Path] | None = None, remove_all: bool = False) -> list[dict]:
    manifest = load_manifest()
    files = manifest.get("files", {})
    if remove_all:
        targets = list(files)
    else:
        targets = []
        for raw in paths or []:
            targets.extend(_match_sources(str(raw), files))
        targets = list(dict.fromkeys(targets))
    if not targets:
        raise FileNotFoundError("В базе знаний нет таких файлов. Посмотрите список: remove_files.py --list")
    base = knowledge_base()
    removed = []
    for source in targets:
        entry = forget_file(source) or {}
        deleted = base.delete_source(source)
        image_dir = entry.get("image_dir") or ""
        if image_dir:
            shutil.rmtree(image_dir, ignore_errors=True)
        removed.append({"source": source, "name": entry.get("name") or Path(source).name, "chunks": deleted})
    return removed


def list_files() -> list[dict]:
    files = load_manifest().get("files", {})
    rows = []
    for source, entry in sorted(files.items(), key=lambda item: item[1].get("name", "").lower()):
        rows.append(
            {
                "source": source,
                "name": entry.get("name") or Path(source).name,
                "chunks": entry.get("chunks", 0),
                "images": entry.get("images", 0),
                "added_at": entry.get("added_at", ""),
            }
        )
    return rows


NO_INFO = "В базе знаний нет информации по этому вопросу."
MIN_HIT_SCORE = 0.55
_WORK_RE = re.compile(
    r"закон|кодекс|стать\w*|норматив|законодатель|правов\w*|юридическ|ответственност|штраф|взят"
    r"|корруп|политик|регламент|положени|инструкц|приказ|документ|пункт\w*|сотрудник|работник"
    r"|коммерческ|тайн|качеств|эколог|обязанност|процедур|стандарт|охрана труда|персональн"
    r"|конфиденц|договор|трудов|дисциплин|этик|комплаенс|полипластик|сертификац|аудит"
    r"|нарушен\w*|санкц|показател|регламент",
    re.IGNORECASE,
)
_SHOW_IMAGE_RE = re.compile(
    r"покаж\w*|посмотреть|как выглядит|схем\w*|рисун\w*|картин\w*|фото|изображен\w*|слайд\w*|иллюстрац",
    re.IGNORECASE,
)


def is_work_question(question: str) -> bool:
    return bool(_WORK_RE.search(question or ""))


def wants_image(question: str) -> bool:
    return bool(_SHOW_IMAGE_RE.search(question or ""))


def _topic_question(question: str) -> str:
    cleaned = _SHOW_IMAGE_RE.sub(" ", question or "")
    cleaned = re.sub(
        r"\b(ответь(?:те)?|пришли(?:те)?|приложи\w*|добавь(?:те)?|покажи(?:те)?)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?!.,;:")
    return cleaned or (question or "").strip()


def _says_no_info(text: str) -> bool:
    compact = re.sub(r"[«»\"']", "", text or "")
    compact = re.sub(r"\s+", " ", compact).strip().rstrip(".")
    return compact.lower() == NO_INFO.rstrip(".").lower()


def _hit_body(hit: Hit) -> str:
    parts = (hit.text or "").split("\n", 1)
    body = parts[1].strip() if len(parts) > 1 else (hit.text or "").strip()
    return body


def _excerpt_answer(hits: list[Hit]) -> str:
    chunks: list[str] = []
    for hit in hits:
        if hit.kind == "image":
            continue
        body = _hit_body(hit)
        if body and body not in chunks:
            chunks.append(body)
        if len(chunks) >= 3:
            break
    return "\n\n".join(chunks)


def _collect_hits(
    question: str,
    top_k: int,
    allowed_documents: list[dict] | None = None,
) -> list[Hit]:
    # Берём запас кандидатов, затем режем по ACL — до генерации ответа.
    raw = knowledge_base().search(_topic_question(question), limit=max(top_k * 8, 32))
    text = [hit for hit in raw if hit.kind != "image" and hit.score >= MIN_HIT_SCORE]
    text = filter_hits(text, allowed_documents)
    return text[:top_k]


def ask(
    question: str,
    top_k: int = TOP_K,
    *,
    allowed_documents: list[dict] | None = None,
    history: list[dict] | None = None,
    access: dict | None = None,
    step_context: dict | None = None,
    address_style: str = "formal",
) -> Answer:
    question = (question or "").strip()
    if not question:
        raise ValueError("Пустой вопрос.")
    tone = analyze_tone(question)
    work = is_work_question(question) and not tone.needs_manager and not tone.needs_hr
    # None = без ACL (локальный клиент); [] / список = только разрешённые документы.
    hits = _collect_hits(question, top_k, allowed_documents) if work else []
    if work and not hits:
        answer = Answer(text=NO_INFO, mode="rag", tone=tone.label, needs_hr=True)
        _save_answer(answer)
        return answer

    images = _select_images(hits, question) if hits else []
    if tone.needs_manager:
        mode = "manager"
    elif tone.needs_hr:
        mode = "hr"
    elif work:
        mode = "rag"
    else:
        mode = "chat"
    context_hits = hits if mode == "rag" else []
    if mode == "manager":
        text = "Понял вас. Передам сигнал руководителю — обсудите ситуацию лично."
    elif mode == "hr":
        text = tone.reply or (HR_NO_INFO if tone.hr_topic else _hr_distress_reply(tone.urgent))
    else:
        prompt = _prompt(
            question,
            context_hits,
            images,
            strict=bool(context_hits),
            history=history,
            access=access,
            step_context=step_context,
            address_style=address_style,
        )
        system = _system(mode, address_style=address_style)
        text = vision().generate(prompt, system=system, max_new_tokens=MAX_NEW_TOKENS)
        if _says_no_info(text):
            excerpt = _excerpt_answer(context_hits)
            if excerpt:
                text = excerpt
            else:
                text = NO_INFO
                images = []
                context_hits = []
    if mode == "rag" and allowed_documents is not None:
        sources = map_sources(context_hits, allowed_documents)
        if not sources:
            text = NO_INFO
            images = []
            context_hits = []
    else:
        sources = _sources(context_hits) if context_hits else []
    answer = Answer(
        text=text,
        images=images if mode == "rag" else [],
        sources=sources,
        mode=mode,
        tone=tone.label,
        needs_hr=(tone.needs_hr and not tone.needs_manager)
        or (mode == "rag" and (not sources or text == NO_INFO)),
        needs_manager=tone.needs_manager,
    )
    _save_answer(answer)
    return answer


def _add_one(path: Path, caption: bool, log, extra_meta: dict | None = None) -> dict:
    parsed = parse_file(path)
    source_hash = hashlib.sha1(parsed.source.encode("utf-8")).hexdigest()[:12]
    image_dir = IMAGE_DIR / source_hash
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    display_name = repair_mojibake(str((extra_meta or {}).get("korsik_title") or parsed.name))
    chunks, image_count, skipped = _chunks_from_blocks(
        parsed.source,
        display_name,
        parsed.blocks,
        image_dir,
        caption=caption,
        log=log,
    )
    if not chunks:
        shutil.rmtree(image_dir, ignore_errors=True)
        raise RuntimeError(f"Из {parsed.name} не получилось извлечь текст, таблицы или картинки.")
    base = knowledge_base()
    korsik_id = str((extra_meta or {}).get("korsik_document_id") or "").strip()
    if korsik_id:
        base.delete_korsik_document(korsik_id)
    base.delete_source(parsed.source)
    records = []
    for index, chunk in enumerate(chunks):
        metadata = {
            "source": chunk.source,
            "source_name": chunk.source_name,
            "kind": chunk.kind,
            "page": chunk.page,
            "image_path": chunk.image_path,
        }
        if extra_meta:
            metadata.update({key: value for key, value in extra_meta.items() if value is not None})
        records.append(
            {
                "id": make_id(chunk.source, chunk.kind, chunk.page, index, chunk.text),
                "text": _indexed_text(chunk),
                "metadata": metadata,
            }
        )
    base.add(records)
    if image_count == 0:
        shutil.rmtree(image_dir, ignore_errors=True)
    remember_file(parsed.source, parsed.name, len(records), image_count, str(image_dir if image_count else ""))
    if skipped:
        log(f"  картинок сверх лимита {MAX_IMAGES_PER_FILE} пропущено: {skipped}")
    return {"source": parsed.source, "name": parsed.name, "chunks": len(records), "images": image_count}


def _chunks_from_blocks(
    source: str,
    name: str,
    blocks: list[Block],
    image_dir: Path,
    caption: bool,
    log,
) -> tuple[list[StoredChunk], int, int]:
    chunks: list[StoredChunk] = []
    saved = 0
    skipped = 0
    seen_digest: dict[str, str] = {}
    caption_cache: dict[str, str] = {}
    for block in blocks:
        if block.kind != "image":
            text = block.text.strip()
            if text:
                chunks.append(StoredChunk(text, source, name, block.kind, block.page))
            continue
        if not block.image:
            continue
        prepared = _normalize_image(block.image)
        if prepared is None:
            continue
        blob, digest = prepared
        path = seen_digest.get(digest)
        if path is None:
            if saved >= MAX_IMAGES_PER_FILE:
                skipped += 1
                continue
            path = str(image_dir / f"{digest}.jpg")
            Path(path).write_bytes(blob)
            seen_digest[digest] = path
            saved += 1
            if caption and digest not in caption_cache:
                log(f"  описываю картинку {saved}")
                try:
                    caption_cache[digest] = vision().caption(path)
                except Exception as exc:  # noqa: BLE001 - файл всё равно должен попасть в базу
                    log(f"  не удалось описать картинку: {exc}")
                    caption_cache[digest] = ""
        description = caption_cache.get(digest, "")
        text = _image_text(name, block.page, block.text, description)
        chunks.append(StoredChunk(text, source, name, "image", block.page, image_path=path))
    return chunks, saved, skipped


def _normalize_image(blob: bytes) -> tuple[bytes, str] | None:
    from io import BytesIO

    try:
        with Image.open(BytesIO(blob)) as source:
            image = source.convert("RGBA").convert("RGB")
            if min(image.size) < MIN_IMAGE_SIDE:
                return None
            scale = min(1.0, 1600 / max(image.size))
            if scale < 1:
                image = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            output = BytesIO()
            image.save(output, format="JPEG", quality=85)
    except Exception:
        return None
    data = output.getvalue()
    if len(data) < 2048:
        return None
    digest = hashlib.sha256(data).hexdigest()[:16]
    return data, digest


def _image_text(name: str, page: int, context: str, description: str) -> str:
    where = f"страница {page}" if page else "файл"
    lines = [f"Изображение, {name}, {where}."]
    if description:
        lines.append(description.strip())
    if context.strip():
        lines.append("Текст рядом: " + context.strip())
    return "\n".join(lines)


def _indexed_text(chunk: StoredChunk) -> str:
    place = f", стр. {chunk.page}" if chunk.page else ""
    header = f"Источник: {chunk.source_name}{place}. Тип: {chunk.kind}."
    return header + "\n" + chunk.text


def _collect(paths: list[str | Path]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            for child in path.rglob("*"):
                if not child.is_file():
                    continue
                if any(part in SKIP_DIRS for part in child.parts):
                    continue
                if child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    found.append(child.resolve())
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
        elif path.is_file():
            raise ValueError(f"Формат не поддерживается: {path.name}")
        else:
            raise FileNotFoundError(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in found:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _match_sources(raw: str, files: dict) -> list[str]:
    candidate = Path(raw).expanduser()
    if candidate.exists():
        resolved = str(candidate.resolve())
        if resolved in files:
            return [resolved]
    needle = Path(raw).name.lower()
    matches = [source for source, entry in files.items() if entry.get("name", "").lower() == needle or Path(source).name.lower() == needle]
    if matches:
        return matches
    partial = [
        source
        for source, entry in files.items()
        if needle in entry.get("name", "").lower() or needle in Path(source).name.lower()
    ]
    if len(partial) == 1:
        return partial
    if len(partial) > 1:
        names = ", ".join(files[source].get("name", source) for source in partial)
        raise FileNotFoundError(f"Несколько файлов подходят под «{raw}»: {names}")
    return []


def _select_images(hits: list[Hit], question: str) -> list[dict]:
    if not wants_image(question):
        return []
    chosen: list[dict] = []
    seen: set[str] = set()

    def add(hit: Hit) -> None:
        path = hit.image_path
        if not path or path in seen or not Path(path).is_file():
            return
        if len(chosen) >= MAX_ANSWER_IMAGES:
            return
        seen.add(path)
        chosen.append(
            {
                "path": path,
                "source": hit.source_name,
                "page": hit.page,
                "caption": _short(hit.text, 400),
            }
        )

    for hit in hits:
        if hit.kind == "image":
            add(hit)
    base = knowledge_base()
    for hit in hits:
        if len(chosen) >= MAX_ANSWER_IMAGES:
            break
        if hit.page <= 0:
            continue
        for image_hit in base.images_for(hit.source, hit.page):
            add(image_hit)
            if len(chosen) >= MAX_ANSWER_IMAGES:
                break
    return chosen


def _sources(hits: list[Hit]) -> list[dict]:
    rows = []
    seen: set[tuple[str, int, str]] = set()
    for hit in hits:
        key = (hit.source, hit.page, hit.kind)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"source": hit.source, "name": hit.source_name, "page": hit.page, "kind": hit.kind})
    return rows


def _prompt(
    question: str,
    hits: list[Hit],
    images: list[dict],
    strict: bool = False,
    history: list[dict] | None = None,
    access: dict | None = None,
    step_context: dict | None = None,
    address_style: str = "formal",
) -> str:
    parts = [f"Вопрос:\n{question}"]
    if history:
        lines = []
        for item in history[-8:]:
            role = str(item.get("role") or "user")
            text = str(item.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text[:500]}")
        if lines:
            parts.append("Недавний диалог:\n" + "\n".join(lines))
    context_bits = []
    if access:
        branch = str(access.get("branch_code") or "").strip()
        category = str(access.get("category") or "").strip()
        if branch:
            context_bits.append(f"площадка {branch}")
        if category:
            context_bits.append(f"категория {category}")
    if step_context:
        stage = str(step_context.get("stage") or "").strip()
        if stage:
            context_bits.append(f"этап {stage}")
    if context_bits:
        parts.append("Контекст сотрудника: " + ", ".join(context_bits) + ".")
    if address_style:
        parts.append("Стиль обращения: " + ("на вы" if address_style == "formal" else "на ты") + ".")
    if hits:
        blocks = [f"[{index}] {hit.text}" for index, hit in enumerate(hits, start=1)]
        parts.append("Фрагменты базы знаний:\n" + "\n\n".join(blocks))
    elif strict:
        parts.append("Фрагменты базы знаний:\nнет")
    if images:
        lines = ["Картинки с этих страниц уже приложены к ответу отдельно. На них можно сослаться одной фразой, без путей к файлам."]
        for index, image in enumerate(images, start=1):
            lines.append(f"{index}. {image['source']}, стр. {image['page']}")
        parts.append("\n".join(lines))
    parts.append("Ответь на вопрос.")
    return "\n\n".join(parts)


def _system(mode: str, address_style: str = "formal") -> str:
    address = "Обращайся на вы." if address_style == "formal" else "Можно обращаться на ты, сохраняя вежливость."
    corporate = (
        "Говори только в корпоративном деловом стиле: спокойно, уважительно, по делу, без сленга, шуток, "
        "эмодзи, панибратства и разговорных сокращений. Ты корпоративный помощник по адаптации, не друг в чате."
    )
    if mode == "chat":
        return (
            f"{corporate} "
            "Отвечай по-русски полно и до конца мысли. "
            "Это общий разговор, не вопрос по документам: не упоминай базу знаний, файлы, страницы и источники. "
            f"{address} Не обрывай ответ на полуслове."
        )
    return (
        f"{corporate} "
        "Это рабочий вопрос. Фрагменты уже отобраны по теме и по правам доступа сотрудника. "
        "Ответь по-русски полно и по делу, до конца мысли: "
        "правила, обязанности, пункты и цифры, которые в них есть. "
        "Не добавляй то, чего во фрагментах нет. Не используй знания из других филиалов или документов. "
        f"{address} Не обрывай ответ на полуслове. "
        "Просьба показать картинки уже выполнена отдельно и не повод отказывать. "
        "Фразу «В базе знаний нет информации по этому вопросу.» напиши только если фрагменты про другую тему, и больше ничего."
    )


def index_uploaded_file(
    path: Path,
    *,
    document_id: str,
    version: int,
    title: str,
    branch_codes: list | None = None,
    categories: list | None = None,
    caption: bool = False,
    log=print,
) -> dict:
    """Индексация файла, загруженного из кабинета HR Корсика."""
    extra = {
        "korsik_document_id": str(document_id),
        "korsik_version": int(version),
        "korsik_title": str(title or path.name),
        "branch_codes": list(branch_codes or []),
        "categories": list(categories or []),
    }
    return _add_one(path, caption=caption, log=log, extra_meta=extra)


def _save_answer(answer: Answer) -> None:
    LAST_ANSWER_JSON.write_text(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [answer.text, ""]
    if answer.images:
        lines.append("Изображения:")
        for image in answer.images:
            relative = _relative(image["path"])
            lines.append(f"- {image['source']}, стр. {image['page']}")
            lines.append("")
            lines.append(f"![]({relative})")
            lines.append("")
    if answer.sources:
        lines.append("Источники:")
        for source in answer.sources:
            page = f", стр. {source['page']}" if source.get("page") else ""
            label = source.get("name") or source.get("document_id") or "document"
            kind = f" ({source['kind']})" if source.get("kind") else ""
            lines.append(f"- {label}{page}{kind}")
    LAST_ANSWER_MD.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _relative(path: str) -> str:
    file = Path(path)
    try:
        return file.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return file.resolve().as_posix()


def _short(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
