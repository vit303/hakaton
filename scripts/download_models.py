"""Скачать Qwen VL и модель эмбеддингов на диск D:."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download  # noqa: E402

from rag.config import EMBED_MODEL, HF_HOME, QWEN_VL_MODEL  # noqa: E402
from rag.store import _local_snapshot  # noqa: E402


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    print(f"Кэш: {HF_HOME}")
    print(f"Скачиваю {QWEN_VL_MODEL}")
    snapshot_download(QWEN_VL_MODEL)
    if _local_snapshot(EMBED_MODEL):
        print(f"{EMBED_MODEL} уже есть на диске, повторно не скачиваю.")
    else:
        print(f"Скачиваю {EMBED_MODEL}")
        snapshot_download(EMBED_MODEL)
    print("Модели на месте.")


if __name__ == "__main__":
    main()
