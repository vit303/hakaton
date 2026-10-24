import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.engine import add_files  # noqa: E402


def main() -> None:
    _utf8()
    parser = argparse.ArgumentParser(description="Добавить файлы или папку в базу знаний.")
    parser.add_argument("paths", nargs="+", help="Файлы или папки: pdf, pptx, docx, xlsx, csv, txt, md, картинки")
    parser.add_argument("--no-caption", action="store_true", help="Не описывать картинки моделью Qwen VL")
    args = parser.parse_args()
    try:
        add_files(args.paths, caption=not args.no_caption)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


if __name__ == "__main__":
    main()
