import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.engine import ask  # noqa: E402


def main() -> None:
    _utf8()
    parser = argparse.ArgumentParser(description="Вопрос к базе знаний.")
    parser.add_argument("question", help="Вопрос")
    parser.add_argument("--top-k", type=int, default=6, help="Сколько фрагментов базы знаний брать")
    parser.add_argument("--json", action="store_true", help="Напечатать ответ целиком в JSON")
    args = parser.parse_args()
    try:
        answer = ask(args.question, top_k=args.top_k)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        return
    print(answer.text)
    if answer.images:
        print("\nКартинки:")
        for image in answer.images:
            print(f"- стр. {image['page']}, {image['source']}")
            print(f"  {image['path']}")
    if answer.sources:
        print("\nИсточники:")
        for source in answer.sources:
            page = f", стр. {source['page']}" if source["page"] else ""
            print(f"- {source['name']}{page} ({source['kind']})")


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
