"""Векторное хранилище базы знаний на уже установленном PyTorch."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rag.config import DATA_DIR, EMBED_MODEL, HF_HOME, MANIFEST_PATH

INDEX_PATH = DATA_DIR / "index.json"


@dataclass
class Hit:
    text: str
    source: str
    source_name: str
    kind: str
    page: int
    image_path: str
    score: float
    korsik_document_id: str = ""
    korsik_version: int | None = None
    korsik_title: str = ""


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def encode(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        import torch
        import torch.nn.functional as functional

        model, tokenizer = self._load()
        use_cls = "bge" in self.model_name.lower()
        vectors: list[list[float]] = []
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            prepared = [prefix + " ".join(text.split()) for text in texts]
        else:
            prepared = [" ".join(text.split()) for text in texts]
        max_length = 256 if is_query else 1024
        for start in range(0, len(prepared), 16):
            batch = tokenizer(
                prepared[start : start + 16],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self._device) for key, value in batch.items()}
            with torch.inference_mode():
                output = model(**batch)
            hidden = output.last_hidden_state
            if use_cls:
                embedding = hidden[:, 0]
            else:
                mask = batch["attention_mask"].unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                embedding = summed / counts
            embedding = functional.normalize(embedding.float(), p=2, dim=1)
            vectors.extend(embedding.cpu().tolist())
        return vectors

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            _fast_cuda()
            source = _local_snapshot(self.model_name) or self.model_name
            self._tokenizer = AutoTokenizer.from_pretrained(source)
            folder = Path(source)
            bins = list(folder.glob("*.bin")) if folder.is_dir() else []
            safetensors = list(folder.glob("*.safetensors")) if folder.is_dir() else []
            # torch 2.5 не открывает .bin через transformers 5, а safetensors у bge-m3 нет.
            if bins and not safetensors:
                config = AutoConfig.from_pretrained(source)
                model = AutoModel.from_config(config)
                state = torch.load(bins[0], map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                self._model = model
            else:
                self._model = AutoModel.from_pretrained(source)
            self._device = _embed_device()
            dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
            try:
                self._model.to(device=self._device, dtype=dtype)
            except RuntimeError:
                self._device = torch.device("cpu")
                self._model.to(device=self._device, dtype=torch.float32)
            self._model.eval()
        return self._model, self._tokenizer


class KnowledgeBase:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self.path = INDEX_PATH
        self._records: list[dict[str, Any]] | None = None
        self._matrix: np.ndarray | None = None

    def add(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        embeddings = self.embedder.encode([item["text"] for item in records], is_query=False)
        stored = self._load()
        for item, vector in zip(records, embeddings):
            stored.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "embedding": vector,
                }
            )
        self._matrix = None
        self._save()

    def delete_source(self, source: str) -> int:
        stored = self._load()
        kept = [item for item in stored if item["metadata"].get("source") != source]
        deleted = len(stored) - len(kept)
        self._records = kept
        self._matrix = None
        self._save()
        return deleted

    def delete_korsik_document(self, document_id: str) -> int:
        document_id = str(document_id or "").strip()
        if not document_id:
            return 0
        stored = self._load()
        kept = [
            item
            for item in stored
            if str(item.get("metadata", {}).get("korsik_document_id") or "") != document_id
        ]
        deleted = len(stored) - len(kept)
        self._records = kept
        self._matrix = None
        self._save()
        return deleted

    def clear(self) -> int:
        stored = self._load()
        deleted = len(stored)
        self._records = []
        self._matrix = None
        self._save()
        return deleted

    def count(self) -> int:
        return len(self._load())

    def search(self, question: str, limit: int = 6) -> list[Hit]:
        stored = self._load()
        if not stored:
            return []
        query = np.asarray(self.embedder.encode([question], is_query=True)[0], dtype=np.float32)
        matrix = self._vectors(stored)
        scores = matrix @ query
        terms = [token for token in _words(question) if len(token) >= 3]
        hits: list[Hit] = []
        for index, item in enumerate(stored):
            text = item["text"]
            meta = item["metadata"]
            lexical = 0.0
            if terms:
                lowered = text.lower()
                lexical = sum(1 for token in terms if token in lowered) / len(terms)
            kind = str(meta.get("kind") or "text")
            if kind == "table":
                lexical += 0.08
            version = meta.get("korsik_version")
            hits.append(
                Hit(
                    text=text,
                    source=str(meta.get("source") or ""),
                    source_name=str(meta.get("source_name") or ""),
                    kind=kind,
                    page=int(meta.get("page") or 0),
                    image_path=str(meta.get("image_path") or ""),
                    score=float(scores[index]) + 0.2 * lexical,
                    korsik_document_id=str(meta.get("korsik_document_id") or ""),
                    korsik_version=int(version) if version is not None else None,
                    korsik_title=str(meta.get("korsik_title") or ""),
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def images_for(self, source: str, page: int) -> list[Hit]:
        hits: list[Hit] = []
        for item in self._load():
            meta = item["metadata"]
            if meta.get("source") != source or int(meta.get("page") or 0) != page or meta.get("kind") != "image":
                continue
            path = str(meta.get("image_path") or "")
            if not path:
                continue
            version = meta.get("korsik_version")
            hits.append(
                Hit(
                    text=item["text"],
                    source=source,
                    source_name=str(meta.get("source_name") or ""),
                    kind="image",
                    page=page,
                    image_path=path,
                    score=0.0,
                    korsik_document_id=str(meta.get("korsik_document_id") or ""),
                    korsik_version=int(version) if version is not None else None,
                    korsik_title=str(meta.get("korsik_title") or ""),
                )
            )
        return hits

    def _vectors(self, stored: list[dict[str, Any]]) -> np.ndarray:
        if self._matrix is not None and len(self._matrix) == len(stored):
            return self._matrix
        if not stored:
            self._matrix = np.zeros((0, 1), dtype=np.float32)
            return self._matrix
        self._matrix = np.asarray([item["embedding"] for item in stored], dtype=np.float32)
        return self._matrix

    def _load(self) -> list[dict[str, Any]]:
        if self._records is None:
            if self.path.exists():
                self._records = json.loads(self.path.read_text(encoding="utf-8"))
            else:
                self._records = []
        return self._records

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._records or [], ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.path)


def make_id(source: str, kind: str, page: int, index: int, text: str) -> str:
    import hashlib

    raw = f"{source}|{kind}|{page}|{index}|{text[:240]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"files": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def remember_file(source: str, name: str, chunks: int, images: int, image_dir: str) -> None:
    manifest = load_manifest()
    manifest.setdefault("files", {})[source] = {
        "name": name,
        "chunks": chunks,
        "images": images,
        "image_dir": image_dir,
        "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_manifest(manifest)


def forget_file(source: str) -> dict[str, Any] | None:
    manifest = load_manifest()
    entry = manifest.get("files", {}).pop(source, None)
    save_manifest(manifest)
    return entry


def weights_ready(model_name: str) -> bool:
    return _local_snapshot(model_name) is not None


def download_in_progress(model_name: str) -> bool:
    folder = HF_HOME / "hub" / ("models--" + model_name.replace("/", "--")) / "blobs"
    return folder.is_dir() and any(folder.glob("*.incomplete"))


def _local_snapshot(model_name: str) -> str | None:
    folder = "models--" + model_name.replace("/", "--")
    roots = [HF_HOME / "hub", Path.home() / ".cache" / "huggingface" / "hub"]
    for root in roots:
        ref = root / folder / "refs" / "main"
        if not ref.is_file():
            continue
        snapshot = root / folder / "snapshots" / ref.read_text(encoding="utf-8").strip()
        weights = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("*.bin"))
        if snapshot.is_dir() and any(path.is_file() and path.stat().st_size > 0 for path in weights):
            return str(snapshot)
    return None


def _fast_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def _embed_device():
    import torch

    if not torch.cuda.is_available():
        return torch.device("cpu")
    free, _total = torch.cuda.mem_get_info()
    # bge-m3 в bf16 занимает около 1.2 ГБ. Если модель ответов уже на карте, запас ещё есть.
    if free < 1_800_000_000:
        return torch.device("cpu")
    return torch.device("cuda")


def _words(text: str) -> list[str]:
    import re

    return re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text.lower())
