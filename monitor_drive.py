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

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

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
                "nextPageToken,files(id,name,mimeType,modifiedTime,webViewLink,owners(displayName),"
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
        raw_files = self._list_files(folder_id)
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
            "file_type": f["mimeType"],
            "modified_time": f["modifiedTime"],
            "file_link": f.get("webViewLink", ""),
        }

    def run_once(self) -> Dict[str, List[str]]:
        folder_id = settings.google_drive_folder_id
        if not folder_id and not settings.google_shared_drive_id:
            raise ValueError("Set DRIVE_FOLDER_ID for folder mode or GOOGLE_SHARED_DRIVE_ID for whole shared drive mode.")

        remote_files = self._list_files(folder_id)
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
