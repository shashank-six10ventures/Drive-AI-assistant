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
            wanted = set([keyword.lower() for keyword in filters["keywords"]])
            filtered = []
            for r in out:
                kws = set([str(keyword).lower() for keyword in json.loads(r.get("keywords", "[]"))])
                haystack = " ".join(
                    [
                        r.get("file_name", ""),
                        r.get("path_text", ""),
                        r.get("summary", ""),
                        r.get("topic", ""),
                    ]
                ).lower()
                if all((keyword in kws) or (keyword in haystack) for keyword in wanted):
                    filtered.append(r)
            out = filtered
        return out

    def _name_search(self, query: str, item_kind: str | None = None, limit: int = 25) -> List[Dict]:
        cleaned = query.strip().strip('"').strip("'")
        if not cleaned:
            return []
        matches = self.indexer.search_by_name(cleaned, item_kind=item_kind, limit=limit)
        exact = [row for row in matches if row.get("file_name", "").lower() == cleaned.lower()]
        partial = [row for row in matches if row.get("file_name", "").lower() != cleaned.lower()]
        return exact + partial

    def _rank_rows(self, query: str, rows: List[Dict], top_k: int) -> List[Dict]:
        if not rows:
            return []
        q_tokens = [token for token in re.split(r"\W+", query.lower()) if token]
        ranked = []
        for row in rows:
            haystack = " ".join(
                [
                    row.get("file_name", ""),
                    row.get("path_text", ""),
                    row.get("summary", ""),
                    row.get("topic", ""),
                    row.get("uploader_name", ""),
                ]
            ).lower()
            lexical_score = sum(1 for token in q_tokens if token in haystack)
            ranked.append((lexical_score, row))
        ranked.sort(key=lambda item: (item[0], item[1].get("updated_at", "")), reverse=True)
        return [row for _, row in ranked[:top_k]]

    def search_with_filters(self, query: str, filters: Dict, top_k: int = 10, base_rows: List[Dict] | None = None) -> List[Dict]:
        if base_rows is not None:
            candidates = base_rows
        else:
            item_kind = (filters or {}).get("item_kind")
            candidates = self.indexer.list_items(item_kind=item_kind)
        merged_filters = {"uploader": None, "year": None, "file_type": None, "item_kind": None, "keywords": []}
        merged_filters.update(filters or {})
        results = self._apply_filters(candidates, merged_filters)
        results = self._rank_rows(query, results, top_k=top_k)
        self.memory.add_turn(query, [r["file_id"] for r in results])
        self.memory.last_results = results
        return results

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        lower = query.lower().strip()
        filters = self._extract_filters(query)
        has_filters = any(
            value not in [None, "", []]
            for value in [filters.get("uploader"), filters.get("year"), filters.get("file_type"), filters.get("item_kind"), filters.get("keywords")]
        )
        filename_match = re.search(
            r"(?:named|file name|filename|search file|find file|open file)\s+([a-zA-Z0-9_.() \-]+)",
            lower,
        )
        if filename_match:
            name_query = filename_match.group(1).strip(" .,!?:;")
            results = self._name_search(name_query, item_kind="file", limit=max(top_k, 50))
            self.memory.add_turn(query, [r["file_id"] for r in results])
            self.memory.last_results = results
            return results
        if "." in query and any(ext in lower for ext in [".csv", ".xlsx", ".xls", ".pdf", ".docx", ".pptx", ".py"]):
            results = self._name_search(query, item_kind=None, limit=max(top_k, 50))
            if results:
                self.memory.add_turn(query, [r["file_id"] for r in results])
                self.memory.last_results = results
                return results
        if any(phrase in lower for phrase in ["show folders", "list folders", "folders available", "what folders"]):
            results = self.indexer.list_items("folder")[: max(top_k, 100)]
            self.memory.add_turn(query, [r["file_id"] for r in results])
            self.memory.last_results = results
            return results
        if any(phrase in lower for phrase in ["all files", "drive files", "list files", "show files", "what are the files"]):
            results = self.indexer.list_items("file")[: max(top_k, 100)]
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
            return self.search_with_filters(query, filters, top_k=top_k, base_rows=base)

        if has_filters:
            return self.search_with_filters(query, filters, top_k=max(top_k, 50), base_rows=None)

        candidates = self.indexer.semantic_search(query, top_k=max(top_k, 50))
        return self.search_with_filters(query, filters, top_k=top_k, base_rows=candidates)
