import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.engine import list_files, remove_files  # noqa: E402


def main() -> None:
    _utf8()
    parser = argparse.ArgumentParser(description="Удалить файлы из базы знаний или показать, что в ней лежит.")
    parser.add_argument("paths", nargs="*", help="Путь или имя файла, который уже добавлен в базу")
    parser.add_argument("--list", action="store_true", help="Показать файлы в базе знаний")
    parser.add_argument("--all", action="store_true", help="Удалить всю базу знаний")
    args = parser.parse_args()
    try:
        if args.list:
            rows = list_files()
            if not rows:
                print("База знаний пуста.")
                return
            for row in rows:
                print(f"{row['name']}  |  фрагментов: {row['chunks']}, картинок: {row['images']}")
                print(f"  {row['source']}")
            return
        if not args.all and not args.paths:
            parser.error("Укажите файлы, --all или --list")
        removed = remove_files(args.paths, remove_all=args.all)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    for row in removed:
        print(f"Удалено: {row['name']} (фрагментов: {row['chunks']})")


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
