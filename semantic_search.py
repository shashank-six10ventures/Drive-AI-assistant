from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

from metadata_indexer import MetadataIndexer


@dataclass
class ConversationMemory:
    history: List[Dict] = field(default_factory=list)
    last_results: List[Dict] = field(default_factory=list)

    def add_turn(self, query: str, result_ids: List[str]) -> None:
        self.history.append({"query": query, "result_ids": result_ids})


class SemanticSearchEngine:
    def __init__(self, indexer: MetadataIndexer):
        self.indexer = indexer
        self.memory = ConversationMemory()

    def _extract_folder_name(self, query: str) -> str | None:
        q = query.lower()
        patterns = [
            r"(?:inside|under|in)\s+folder\s+([a-z0-9 _./-]+)",
            r"folder\s+([a-z0-9 _./-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                return match.group(1).strip(" .,!?:;")
        return None

    def _extract_filters(self, query: str) -> Dict:
        q = query.lower()
        filters = {"uploader": None, "year": None, "file_type": None, "item_kind": None, "keywords": []}
        m = re.search(r"uploaded by\s+([a-zA-Z0-9 _-]+)", q)
        if m:
            # Keep uploader filter clean so "uploaded by Amber?" still works.
            filters["uploader"] = m.group(1).strip(" .,!?:;")
        year = re.search(r"\b(20\d{2})\b", q)
        if year:
            filters["year"] = year.group(1)
        for token in ["sales", "revenue", "profit", "marketing", "finance"]:
            if token in q:
                filters["keywords"].append(token)
        if "folder" in q or "folders" in q:
            filters["item_kind"] = "folder"
        if "file" in q or "files" in q or "dataset" in q or "datasets" in q:
            filters["item_kind"] = "file"
        return filters

    def _apply_filters(self, rows: List[Dict], filters: Dict) -> List[Dict]:
        out = rows
        if filters.get("uploader"):
            u = str(filters["uploader"]).lower()
            out = [r for r in out if u in r.get("uploader_name", "").lower()]
        if filters.get("year"):
            y = filters["year"]
            out = [r for r in out if y in r.get("modified_time", "")]
        if filters.get("file_type"):
            ft = filters["file_type"].lower()
            out = [r for r in out if ft in r.get("file_type", "").lower() or ft in r.get("file_name", "").lower()]
        if filters.get("item_kind"):
            kind = filters["item_kind"]
            out = [r for r in out if r.get("item_kind", "file") == kind]
        if filters.get("keywords"):
            wanted = set(filters["keywords"])
            filtered = []
            for r in out:
                kws = set(json.loads(r.get("keywords", "[]")))
                if wanted.intersection(kws):
                    filtered.append(r)
            out = filtered
        return out

    def search_with_filters(self, query: str, filters: Dict, top_k: int = 10, base_rows: List[Dict] | None = None) -> List[Dict]:
        candidates = base_rows if base_rows is not None else self.indexer.semantic_search(query, top_k=top_k)
        merged_filters = {"uploader": None, "year": None, "file_type": None, "item_kind": None, "keywords": []}
        merged_filters.update(filters or {})
        results = self._apply_filters(candidates, merged_filters)
        self.memory.add_turn(query, [r["file_id"] for r in results])
        self.memory.last_results = results
        return results

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        lower = query.lower().strip()
        if any(phrase in lower for phrase in ["show folders", "list folders", "folders available", "what folders"]):
            results = self.indexer.list_items("folder")[:top_k]
            self.memory.add_turn(query, [r["file_id"] for r in results])
            self.memory.last_results = results
            return results
        if any(phrase in lower for phrase in ["all files", "drive files", "list files", "show files", "what are the files"]):
            results = self.indexer.list_items("file")[:top_k]
            self.memory.add_turn(query, [r["file_id"] for r in results])
            self.memory.last_results = results
            return results
        if any(phrase in lower for phrase in ["show contents", "inside folder", "in folder", "under folder"]):
            folder_name = self._extract_folder_name(query)
            if folder_name:
                folders = self.indexer.search_by_name(folder_name, item_kind="folder", limit=3)
                if folders:
                    results = self.indexer.list_children(folders[0]["file_id"])[:top_k]
                    self.memory.add_turn(query, [r["file_id"] for r in results])
                    self.memory.last_results = results
                    return results

        if lower.startswith("from these") or lower.startswith("compare") or "across them" in lower:
            base = self.memory.last_results
            if not base:
                return []
            filters = self._extract_filters(query)
            return self.search_with_filters(query, filters, top_k=top_k, base_rows=base)

        candidates = self.indexer.semantic_search(query, top_k=top_k)
        filters = self._extract_filters(query)
        return self.search_with_filters(query, filters, top_k=top_k, base_rows=candidates)
