"""Пути, модели и ключи. Импортируйте этот модуль до huggingface и chromadb."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
IMAGE_DIR = DATA_DIR / "images"
MANIFEST_PATH = DATA_DIR / "manifest.json"
LAST_ANSWER_JSON = DATA_DIR / "last_answer.json"
LAST_ANSWER_MD = DATA_DIR / "last_answer.md"

CACHE_DIR = ROOT / ".cache"
HF_HOME = CACHE_DIR / "huggingface"

load_dotenv(ROOT / ".env")

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# В torch 2.5 на Windows этот режим аллокатора роняет процесс.
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

for path in (DATA_DIR, CHROMA_DIR, IMAGE_DIR, HF_HOME):
    path.mkdir(parents=True, exist_ok=True)

# 4B в bf16 занимает около 8 ГБ и оставляет запас под эмбеддер на RTX 4060 Ti 16 ГБ.
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
HR_CONTACT = os.getenv("HR_CONTACT", "HR").strip() or "HR"

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
TABLE_CHUNK_SIZE = int(os.getenv("RAG_TABLE_CHUNK_SIZE", "1400"))
MAX_IMAGES_PER_FILE = int(os.getenv("RAG_MAX_IMAGES", "40"))
MIN_IMAGE_SIDE = int(os.getenv("RAG_MIN_IMAGE_SIDE", "80"))
MAX_IMAGE_SIDE = int(os.getenv("RAG_MAX_IMAGE_SIDE", "1024"))
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "1024"))
MAX_ANSWER_IMAGES = int(os.getenv("RAG_MAX_ANSWER_IMAGES", "3"))

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".pptx",
    ".docx",
    ".xlsx",
    ".csv",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

SKIP_DIRS = {".cache", "data", ".git", "__pycache__", ".venv", "venv"}
