from __future__ import annotations

import io
import json
import logging
import re
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.oauth2.credentials import Credentials as UserCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow

from config import CACHE_DIR, settings
from file_analyzer import analyze_file
from metadata_indexer import MetadataIndexer


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SCOPES = ["https://www.googleapis.com/auth/drive"]

GOOGLE_EXPORT_MAP = {
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
}


class DriveMonitor:
    def __init__(self):
        self.indexer = MetadataIndexer()
        self.drive = self._build_drive_client()

    def _build_drive_client(self):
        creds = self._build_credentials()
        return build("drive", "v3", credentials=creds)

    def _build_credentials(self):
        # Supports service account (server) and OAuth user flow (local desktop usage).
        if settings.drive_auth_mode == "oauth":
            token_path = Path(settings.google_oauth_token_file)
            client_path = Path(settings.google_oauth_client_file)
            if not client_path.exists() and not token_path.exists():
                raise FileNotFoundError(
                    f"OAuth credentials file not found: {settings.google_oauth_client_file}"
                )
            creds = None
            if token_path.exists():
                creds = UserCredentials.from_authorized_user_file(str(token_path), SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            if not creds or not creds.valid:
                flow = InstalledAppFlow.from_client_secrets_file(str(client_path), SCOPES)
                creds = flow.run_local_server(port=0)
                token_path.write_text(creds.to_json(), encoding="utf-8")
            return creds
        if settings.google_service_account_json.strip():
            service_account_info = self._parse_service_account_json(settings.google_service_account_json)
            return Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
        if not Path(settings.google_service_account_file).exists():
            raise FileNotFoundError(
                f"Service account file not found: {settings.google_service_account_file}"
            )
        return Credentials.from_service_account_file(settings.google_service_account_file, scopes=SCOPES)

    def _parse_service_account_json(self, raw_json: str) -> Dict:
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            # Streamlit secrets sometimes end up with literal newlines inside private_key.
            fixed_json = re.sub(
                r'("private_key"\s*:\s*")(.*?)("\s*,)',
                lambda match: match.group(1)
                + match.group(2).replace("\\", "\\\\").replace("\r", "").replace("\n", "\\n")
                + match.group(3),
                raw_json,
                flags=re.DOTALL,
            )
            return json.loads(fixed_json)

    def extract_file_metadata(self, drive_file: Dict) -> Dict:
        # Normalizes raw Drive payload into metadata used by the index.
        return self._normalize_metadata(drive_file)

    def _list_files(self, folder_id: str | None = None) -> List[Dict]:
        is_whole_shared_drive_scan = bool(settings.google_shared_drive_id and not folder_id)
        q = "trashed=false" if is_whole_shared_drive_scan else f"'{folder_id}' in parents and trashed=false"
        params: Dict = {
            "q": q,
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
            "fields": (
                "nextPageToken,files(id,name,mimeType,modifiedTime,webViewLink,parents,owners(displayName),"
                "lastModifyingUser(displayName))"
            ),
            "pageSize": 1000,
        }
        if settings.google_shared_drive_id:
            params["driveId"] = settings.google_shared_drive_id
            params["corpora"] = "drive"
        elif is_whole_shared_drive_scan:
            raise ValueError("GOOGLE_SHARED_DRIVE_ID is required for whole shared drive scanning.")
        all_files: List[Dict] = []
        page_token = None
        while True:
            if page_token:
                params["pageToken"] = page_token
            response = self.drive.files().list(**params).execute()
            all_files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return all_files

    def _list_tree_items(self, root_folder_id: str) -> List[Dict]:
        seen_ids = set()
        ordered_items: List[Dict] = []
        pending_folders = [root_folder_id]

        while pending_folders:
            current_folder_id = pending_folders.pop(0)
            for item in self._list_files(current_folder_id):
                if item["id"] in seen_ids:
                    continue
                seen_ids.add(item["id"])
                ordered_items.append(item)
                if item.get("mimeType") == "application/vnd.google-apps.folder":
                    pending_folders.append(item["id"])
        return ordered_items

    def _build_path_lookup(self, root_folder_id: str, raw_items: List[Dict]) -> Dict[str, str]:
        path_lookup: Dict[str, str] = {}
        children_by_parent: Dict[str, List[Dict]] = {}
        for item in raw_items:
            for parent_id in item.get("parents", []):
                children_by_parent.setdefault(parent_id, []).append(item)

        queue: List[Tuple[str, str]] = [(root_folder_id, "")]
        while queue:
            current_parent_id, current_path = queue.pop(0)
            for child in sorted(children_by_parent.get(current_parent_id, []), key=lambda row: row.get("name", "").lower()):
                child_path = f"{current_path}/{child['name']}".strip("/")
                path_lookup[child["id"]] = child_path
                if child.get("mimeType") == "application/vnd.google-apps.folder":
                    queue.append((child["id"], child_path))
        return path_lookup

    def get_drive_files(self, folder_id: str | None = None) -> List[Dict]:
        """
        Fetch normalized metadata for all files inside the given Drive folder,
        or the entire shared drive when folder_id is omitted and GOOGLE_SHARED_DRIVE_ID is set.
        Returns items with:
        - file_id
        - file_name
        - file_type
        - modified_time
        - file_link
        - uploader_name
        """
        raw_files = self._list_tree_items(folder_id) if folder_id else self._list_files(folder_id)
        if folder_id:
            path_lookup = self._build_path_lookup(folder_id, raw_files)
            for raw in raw_files:
                raw["pathText"] = path_lookup.get(raw["id"], raw.get("name", ""))
        return [self.extract_file_metadata(f) for f in raw_files]

    def _download_file_bytes(self, file_id: str, file_name: str, mime_type: str) -> Tuple[bytes, str, str]:
        if mime_type in GOOGLE_EXPORT_MAP:
            export_mime, ext = GOOGLE_EXPORT_MAP[mime_type]
            request = self.drive.files().export_media(fileId=file_id, mimeType=export_mime)
            resolved_name = f"{Path(file_name).stem}{ext}"
            resolved_mime = export_mime
        else:
            request = self.drive.files().get_media(fileId=file_id, supportsAllDrives=True)
            resolved_name = file_name
            resolved_mime = mime_type

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue(), resolved_name, resolved_mime

    def _save_local_copy(self, file_id: str, file_name: str, file_bytes: bytes) -> str:
        out_path = CACHE_DIR / f"{file_id}_{file_name}"
        out_path.write_bytes(file_bytes)
        return str(out_path)

    def _notify(self, message: str) -> None:
        self._notify_slack(message)
        self._notify_email("Drive AI Assistant Alert", message)

    def _notify_slack(self, message: str) -> None:
        if not settings.slack_webhook_url:
            return
        try:
            requests.post(settings.slack_webhook_url, json={"text": message}, timeout=20)
        except Exception as exc:
            logging.warning("Slack notification failed: %s", exc)

    def _notify_email(self, subject: str, body: str) -> None:
        if not all(
            [settings.smtp_host, settings.smtp_username, settings.smtp_password, settings.email_to, settings.email_from]
        ):
            return
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = settings.email_from
            msg["To"] = settings.email_to
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                server.starttls()
                server.login(settings.smtp_username, settings.smtp_password)
                server.sendmail(settings.email_from, [settings.email_to], msg.as_string())
        except Exception as exc:
            logging.warning("Email notification failed: %s", exc)

    def _normalize_metadata(self, f: Dict) -> Dict:
        uploader = (
            (f.get("owners", [{}])[0].get("displayName", "Unknown"))
            or f.get("lastModifyingUser", {}).get("displayName")
            or "Unknown"
        )
        return {
            "file_id": f["id"],
            "file_name": f["name"],
            "uploader_name": uploader,
            "item_kind": "folder" if f.get("mimeType") == "application/vnd.google-apps.folder" else "file",
            "file_type": f["mimeType"],
            "modified_time": f["modifiedTime"],
            "file_link": f.get("webViewLink", ""),
            "parent_ids": f.get("parents", []),
            "path_text": f.get("pathText", f.get("name", "")),
        }

    def suggest_clean_name(self, original_name: str, item_kind: str = "file") -> str:
        suffix = ""
        stem = original_name
        if item_kind == "file":
            suffix = Path(original_name).suffix
            stem = Path(original_name).stem

        stem = stem.replace("_", " ").replace("-", " ")
        stem = re.sub(r"\s+", " ", stem).strip()
        stem = re.sub(r"[^\w\s().,&]", "", stem)
        words = []
        for word in stem.split():
            if word.isupper() or word.isdigit():
                words.append(word)
            elif len(word) <= 3 and word.lower() in {"sku", "api", "pdf", "csv", "pnl"}:
                words.append(word.upper())
            else:
                words.append(word.capitalize())
        cleaned = " ".join(words).strip(" ._")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return f"{cleaned}{suffix}" if cleaned else original_name

    def preview_clean_names(self, items: List[Dict]) -> List[Dict]:
        preview = []
        for item in items:
            current_name = item.get("file_name") or item.get("name", "")
            suggested_name = self.suggest_clean_name(current_name, item.get("item_kind", "file"))
            preview.append(
                {
                    "file_id": item["file_id"],
                    "current_name": current_name,
                    "suggested_name": suggested_name,
                    "item_kind": item.get("item_kind", "file"),
                    "path_text": item.get("path_text", ""),
                    "will_change": current_name != suggested_name,
                }
            )
        return preview

    def rename_drive_item(self, file_id: str, new_name: str) -> Dict:
        updated = (
            self.drive.files()
            .update(fileId=file_id, body={"name": new_name}, supportsAllDrives=True, fields="id,name,modifiedTime")
            .execute()
        )
        return updated

    def apply_rename_plan(self, rename_plan: List[Dict]) -> List[Dict]:
        applied = []
        for item in rename_plan:
            if not item.get("will_change"):
                continue
            try:
                updated = self.rename_drive_item(item["file_id"], item["suggested_name"])
                applied.append(updated)
            except Exception as exc:
                logging.warning("Failed to rename %s: %s", item["file_id"], exc)
        return applied

    def run_once(self) -> Dict[str, List[str]]:
        folder_id = settings.google_drive_folder_id
        if not folder_id and not settings.google_shared_drive_id:
            raise ValueError("Set DRIVE_FOLDER_ID for folder mode or GOOGLE_SHARED_DRIVE_ID for whole shared drive mode.")

        remote_files = self._list_tree_items(folder_id) if folder_id else self._list_files(None)
        path_lookup = self._build_path_lookup(folder_id, remote_files) if folder_id else {}
        for raw in remote_files:
            raw["pathText"] = path_lookup.get(raw["id"], raw.get("name", ""))
        remote_by_id = {f["id"]: f for f in remote_files}
        prior = self.indexer.get_monitor_state()
        current = {f["id"]: f["modifiedTime"] for f in remote_files}

        new_ids = [fid for fid in current if fid not in prior]
        modified_ids = [fid for fid in current if fid in prior and prior[fid] != current[fid]]
        deleted_ids = [fid for fid in prior if fid not in current]

        for fid in new_ids + modified_ids:
            raw = remote_by_id[fid]
            metadata = self.extract_file_metadata(raw)
            try:
                if metadata["item_kind"] == "folder":
                    analysis = {
                        "summary": f"Drive folder: {metadata['file_name']}",
                        "keywords": ["folder"],
                        "topic": "folder",
                        "dataset_type": "folder",
                        "columns": [],
                        "num_rows": 0,
                        "sample_data": [],
                        "text_content": metadata["file_name"],
                    }
                    self.indexer.index_file(metadata, analysis, local_path="")
                    self.indexer.update_monitor_state(fid, metadata["modified_time"])
                    continue
                file_bytes, resolved_name, resolved_mime = self._download_file_bytes(
                    file_id=fid, file_name=metadata["file_name"], mime_type=metadata["file_type"]
                )
                analysis = analyze_file(resolved_name, resolved_mime, file_bytes)
                local_path = self._save_local_copy(fid, resolved_name, file_bytes)
                metadata["file_name"] = resolved_name
                metadata["file_type"] = resolved_mime
                self.indexer.index_file(metadata, analysis, local_path=local_path)
                self.indexer.update_monitor_state(fid, metadata["modified_time"])
            except Exception as exc:
                logging.exception("Failed to process file %s: %s", fid, exc)

        for fid in deleted_ids:
            self.indexer.remove_file(fid)

        changes = {"new": new_ids, "modified": modified_ids, "deleted": deleted_ids}
        if any(changes.values()):
            msg = (
                "Drive changes detected\n"
                f"New: {len(new_ids)}\nModified: {len(modified_ids)}\nDeleted: {len(deleted_ids)}"
            )
            self._notify(msg)
            logging.info(msg)
        else:
            logging.info("No changes detected.")
        return changes

    def run_forever(self):
        logging.info("Drive monitor started. Polling every %s seconds.", settings.monitor_interval_seconds)
        while True:
            try:
                self.run_once()
            except Exception as exc:
                logging.exception("Monitor cycle failed: %s", exc)
            time.sleep(settings.monitor_interval_seconds)


if __name__ == "__main__":
    DriveMonitor().run_forever()
