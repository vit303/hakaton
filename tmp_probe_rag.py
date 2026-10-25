"""Probe search/ACL without loading the VL model (GPU already held by ml_service)."""

from rag.access import filter_hits, normalize_title, titles_match
from rag.store import KnowledgeBase, load_manifest

q = "Какие обязанности описаны в антикоррупционной политике?"
kb = KnowledgeBase()
raw = kb.search(q, limit=16)
print("raw_hits", len(raw))
for h in raw[:12]:
    meta = {
        "source_name": h.source_name,
        "page": h.page,
        "kind": h.kind,
        "score": round(h.score, 3),
        "korsik_document_id": getattr(h, "korsik_document_id", "") or "",
        "korsik_version": getattr(h, "korsik_version", None),
    }
    print(meta)
    print(" ", (h.text or "")[:120].replace("\n", " | "))

print("\n--- manifest names ---")
for _src, entry in load_manifest().get("files", {}).items():
    name = entry.get("name") or ""
    print(repr(name), "norm=", repr(normalize_title(name)), "chunks=", entry.get("chunks"))

target = "Антикоррупционная политика"
print("\n--- title match vs", repr(target), "norm=", repr(normalize_title(target)))
for h in raw:
    print(titles_match(h.source_name, target), repr(h.source_name), "->", repr(normalize_title(h.source_name)))

allowed = [{"id": "doc-anti", "title": target, "version": 1}]
text_hits = [h for h in raw if h.kind != "image" and h.score >= 0.55]
print("\ntext_hits>=0.55", len(text_hits), "filtered", len(filter_hits(text_hits, allowed)))
print("filtered empty ACL", len(filter_hits(text_hits, [])))
print("filtered None ACL", len(filter_hits(text_hits, None)))
