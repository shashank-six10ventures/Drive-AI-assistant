from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import CACHE_DIR, DB_PATH, settings
from file_analyzer import analyze_file


@dataclass
class IndexedFile:
    file_id: str
    file_name: str
    uploader_name: str
    file_type: str
    modified_time: str
    file_link: str
    summary: str
    keywords: List[str]
    topic: str
    dataset_type: str
    columns: List[str]
    num_rows: int
    sample_data: List[Dict]
    text_content: str
    local_path: str


class MetadataIndexer:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.model = SentenceTransformer(settings.embedding_model)
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                file_name TEXT,
                uploader_name TEXT,
                file_type TEXT,
                modified_time TEXT,
                file_link TEXT,
                summary TEXT,
                keywords TEXT,
                topic TEXT,
                dataset_type TEXT,
                columns_json TEXT,
                num_rows INTEGER,
                sample_data_json TEXT,
                text_content TEXT,
                local_path TEXT,
                embedding_json TEXT,
                updated_at TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS monitor_state (
                file_id TEXT PRIMARY KEY,
                modified_time TEXT,
                deleted INTEGER DEFAULT 0
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_files_uploader ON files(uploader_name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_files_modified ON files(modified_time)")
        self.conn.commit()

    def index_file(self, metadata: Dict, analysis: Dict, local_path: str) -> None:
        combined_text = " ".join(
            [
                metadata.get("file_name", ""),
                analysis.get("summary", ""),
                " ".join(analysis.get("keywords", [])),
                analysis.get("text_content", "")[:5000],
            ]
        )
        embedding = self.model.encode(combined_text).tolist()
        self.conn.execute(
            """
            INSERT INTO files (
                file_id, file_name, uploader_name, file_type, modified_time, file_link,
                summary, keywords, topic, dataset_type, columns_json, num_rows,
                sample_data_json, text_content, local_path, embedding_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_id) DO UPDATE SET
                file_name=excluded.file_name,
                uploader_name=excluded.uploader_name,
                file_type=excluded.file_type,
                modified_time=excluded.modified_time,
                file_link=excluded.file_link,
                summary=excluded.summary,
                keywords=excluded.keywords,
                topic=excluded.topic,
                dataset_type=excluded.dataset_type,
                columns_json=excluded.columns_json,
                num_rows=excluded.num_rows,
                sample_data_json=excluded.sample_data_json,
                text_content=excluded.text_content,
                local_path=excluded.local_path,
                embedding_json=excluded.embedding_json,
                updated_at=excluded.updated_at
            """,
            (
                metadata["file_id"],
                metadata.get("file_name", ""),
                metadata.get("uploader_name", ""),
                metadata.get("file_type", ""),
                metadata.get("modified_time", ""),
                metadata.get("file_link", ""),
                analysis.get("summary", ""),
                json.dumps(analysis.get("keywords", [])),
                analysis.get("topic", "general"),
                analysis.get("dataset_type", "general"),
                json.dumps(analysis.get("columns", [])),
                int(analysis.get("num_rows", 0)),
                json.dumps(analysis.get("sample_data", [])),
                analysis.get("text_content", ""),
                local_path,
                json.dumps(embedding),
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    def remove_file(self, file_id: str) -> None:
        self.conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
        self.conn.execute(
            "INSERT INTO monitor_state(file_id, modified_time, deleted) VALUES (?, ?, 1) "
            "ON CONFLICT(file_id) DO UPDATE SET deleted=1",
            (file_id, datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def update_monitor_state(self, file_id: str, modified_time: str) -> None:
        self.conn.execute(
            """
            INSERT INTO monitor_state(file_id, modified_time, deleted)
            VALUES (?, ?, 0)
            ON CONFLICT(file_id) DO UPDATE SET modified_time=excluded.modified_time, deleted=0
            """,
            (file_id, modified_time),
        )
        self.conn.commit()

    def get_monitor_state(self) -> Dict[str, str]:
        rows = self.conn.execute(
            "SELECT file_id, modified_time FROM monitor_state WHERE deleted=0"
        ).fetchall()
        return {row["file_id"]: row["modified_time"] for row in rows}

    def list_files(self) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM files ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def get_all_records(self) -> List[Dict]:
        # Compatibility helper used by seed logic.
        return self.list_files()

    def insert_record(self, item: Dict) -> None:
        # Create a real local demo file so analytics and previews still work.
        file_name = item.get("file_name", "dummy_file.txt")
        file_id = f"dummy_{Path(file_name).stem.lower().replace(' ', '_')}"
        file_path = self._write_dummy_file(file_id, file_name)
        file_bytes = file_path.read_bytes()
        metadata = {
            "file_id": file_id,
            "file_name": file_name,
            "uploader_name": item.get("uploader", "Unknown"),
            "file_type": "text/csv" if file_name.lower().endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "modified_time": datetime.utcnow().isoformat() + "Z",
            "file_link": item.get("link", ""),
        }
        analysis = analyze_file(file_name, metadata["file_type"], file_bytes)
        self.index_file(metadata, analysis, local_path=str(file_path))
        self.update_monitor_state(file_id, metadata["modified_time"])

    def _write_dummy_file(self, file_id: str, file_name: str) -> Path:
        dummy_dir = CACHE_DIR / "dummy"
        dummy_dir.mkdir(parents=True, exist_ok=True)
        file_path = dummy_dir / f"{file_id}_{file_name}"

        if file_name == "sales_2025.xlsx":
            df = pd.DataFrame(
                {
                    "month": ["Jan", "Feb", "Mar", "Apr"],
                    "revenue": [120000, 135000, 142000, 150000],
                    "sales": [320, 350, 372, 395],
                }
            )
            df.to_excel(file_path, index=False)
            return file_path

        if file_name == "profit_analysis.xlsx":
            df = pd.DataFrame(
                {
                    "quarter": ["Q1", "Q2", "Q3", "Q4"],
                    "profit": [22000, 26000, 25000, 31000],
                    "margin": [0.18, 0.20, 0.19, 0.22],
                }
            )
            df.to_excel(file_path, index=False)
            return file_path

        if file_name == "marketing_data.csv":
            df = pd.DataFrame(
                {
                    "campaign": ["Launch A", "Launch B", "Festival Push", "Retargeting"],
                    "leads": [540, 620, 710, 480],
                    "spend": [12000, 15000, 18000, 9000],
                }
            )
            df.to_csv(file_path, index=False)
            return file_path

        file_path.write_text("demo file", encoding="utf-8")
        return file_path

    def get_file(self, file_id: str) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        return dict(row) if row else None

    def semantic_search(self, query: str, top_k: int = 8) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM files").fetchall()
        if not rows:
            return []
        q_emb = self.model.encode(query)
        scored = []
        for row in rows:
            emb = np.array(json.loads(row["embedding_json"]))
            sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-9))
            scored.append((sim, dict(row)))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, item in scored[:top_k]:
            item["score"] = round(sim, 4)
            results.append(item)
        return results

    def count_files(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS c FROM files").fetchone()["c"])

    def count_real_files(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM files WHERE file_id NOT LIKE 'dummy_%'").fetchone()
        return int(row["c"])

    def remove_dummy_data(self) -> int:
        ids = [
            row["file_id"]
            for row in self.conn.execute("SELECT file_id FROM files WHERE file_id LIKE 'dummy_%'").fetchall()
        ]
        if not ids:
            return 0
        self.conn.executemany("DELETE FROM files WHERE file_id = ?", [(fid,) for fid in ids])
        self.conn.executemany("DELETE FROM monitor_state WHERE file_id = ?", [(fid,) for fid in ids])
        self.conn.commit()
        return len(ids)

    def ensure_dummy_data_ready(self) -> int:
        # Repairs stale demo state where dummy rows exist but local files are missing.
        dummy_rows = self.conn.execute(
            "SELECT file_id, local_path FROM files WHERE file_id LIKE 'dummy_%'"
        ).fetchall()
        if not dummy_rows:
            seeded = self.seed_dummy_data_if_empty()
            return 3 if seeded else 0

        valid_local_files = 0
        for row in dummy_rows:
            local_path = row["local_path"] or ""
            if local_path and Path(local_path).exists():
                valid_local_files += 1

        if valid_local_files > 0:
            return valid_local_files

        self.remove_dummy_data()
        self.seed_dummy_data_if_empty()
        repaired_rows = self.conn.execute(
            "SELECT COUNT(*) AS c FROM files WHERE file_id LIKE 'dummy_%'"
        ).fetchone()["c"]
        return int(repaired_rows)

    def _build_dummy_datasets(self) -> List[Dict]:
        # Temporary sample files for local testing when Drive data is unavailable.
        datasets = [
            {
                "file_id": "dummy_sales_2025",
                "file_name": "sales_2025.xlsx",
                "uploader_name": "Amber",
                "file_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "modified_time": "2025-12-31T10:00:00Z",
                "file_link": "https://example.com/sales_2025.xlsx",
                "df": pd.DataFrame(
                    {
                        "month": ["Jan", "Feb", "Mar", "Apr"],
                        "revenue": [120000, 135000, 142000, 150000],
                        "sales": [320, 350, 372, 395],
                    }
                ),
                "ext": ".xlsx",
            },
            {
                "file_id": "dummy_profit_analysis",
                "file_name": "profit_analysis.xlsx",
                "uploader_name": "Neel",
                "file_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "modified_time": "2025-11-15T09:30:00Z",
                "file_link": "https://example.com/profit_analysis.xlsx",
                "df": pd.DataFrame(
                    {
                        "quarter": ["Q1", "Q2", "Q3", "Q4"],
                        "profit": [22000, 26000, 25000, 31000],
                        "margin": [0.18, 0.2, 0.19, 0.22],
                    }
                ),
                "ext": ".xlsx",
            },
            {
                "file_id": "dummy_marketing_data",
                "file_name": "marketing_data.csv",
                "uploader_name": "Madhav",
                "file_type": "text/csv",
                "modified_time": "2025-10-05T08:00:00Z",
                "file_link": "https://example.com/marketing_data.csv",
                "df": pd.DataFrame(
                    {
                        "campaign": ["Launch A", "Launch B", "Festival Push", "Retargeting"],
                        "leads": [540, 620, 710, 480],
                        "spend": [12000, 15000, 18000, 9000],
                    }
                ),
                "ext": ".csv",
            },
        ]
        return datasets

    def seed_dummy_data_if_empty(self):
        """
        If database is empty, insert dummy test data.
        """
        records = self.get_all_records()
        if len(records) > 0:
            return False

        dummy_data = [
            {
                "file_name": "sales_2025.xlsx",
                "uploader": "Amber",
                "keywords": "sales revenue",
                "topic": "sales",
                "link": "",
            },
            {
                "file_name": "profit_analysis.xlsx",
                "uploader": "Neel",
                "keywords": "profit finance",
                "topic": "finance",
                "link": "",
            },
            {
                "file_name": "marketing_data.csv",
                "uploader": "Madhav",
                "keywords": "marketing campaign",
                "topic": "marketing",
                "link": "",
            },
        ]

        for item in dummy_data:
            self.insert_record(item)

        return True
