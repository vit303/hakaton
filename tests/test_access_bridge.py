"""Проверки ACL/сопоставления источников без загрузки GPU-моделей."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.access import filter_hits, map_sources, normalize_title, resolve_allowed, titles_match
from rag.engine import _parse_clusters


class AccessBridgeTests(unittest.TestCase):
    def test_titles_match_exact_only(self):
        self.assertTrue(titles_match("Политика доступного филиала", "политика доступного филиала"))
        self.assertFalse(titles_match("Политика секретного филиала", "Политика доступного филиала"))
        self.assertFalse(titles_match("филиала", "Политика доступного филиала"))

    def test_resolve_by_document_id(self):
        allowed = [{"id": "doc-a", "title": "Политика доступного филиала", "version": 2}]
        hit = {
            "source_name": "Политика секретного филиала.pdf",
            "korsik_document_id": "doc-a",
            "korsik_version": 2,
            "page": 3,
        }
        matched = resolve_allowed(hit, allowed)
        self.assertIsNotNone(matched)
        self.assertEqual(matched["id"], "doc-a")

    def test_resolve_rejects_fuzzy_title(self):
        allowed = [{"id": "doc-a", "title": "Политика доступного филиала", "version": 1}]
        hit = {"source_name": "Политика секретного филиала", "page": 1}
        self.assertIsNone(resolve_allowed(hit, allowed))

    def test_filter_and_map(self):
        allowed = [
            {"id": "open", "title": "Политика доступного филиала", "version": 1},
            {"id": "secret", "title": "Политика секретного филиала", "version": 1},
        ]
        hits = [
            {"source_name": "Политика секретного филиала", "page": 2},
            {"source_name": "Политика доступного филиала", "page": 4},
        ]
        # Сотруднику разрешён только open
        visible = filter_hits(hits, [allowed[0]])
        self.assertEqual(len(visible), 1)
        self.assertEqual(normalize_title(visible[0]["source_name"]), normalize_title(allowed[0]["title"]))
        sources = map_sources(visible, [allowed[0]])
        self.assertEqual(sources, [{"document_id": "open", "version": 1, "page": 4}])

    def test_empty_allowed_filters_everything(self):
        hits = [{"source_name": "Любой документ", "page": 1}]
        self.assertEqual(filter_hits(hits, []), [])

    def test_uncertain_cluster_not_ok(self):
        parsed = _parse_clusters("блабла неизвестно")
        self.assertEqual(parsed[0]["id"], "classification_uncertain")
        self.assertNotEqual(parsed[0]["id"], "ok")


if __name__ == "__main__":
    unittest.main()
