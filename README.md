# Drive AI Assistant

Production-ready AI-powered system for Google Drive monitoring, intelligent file analysis, semantic retrieval, conversational follow-up, analytics, file-renaming workflows, Amazon comparisons, and Streamlit dashboards.

## Implemented Modules

- `monitor_drive.py`: polls Google Drive folder, detects new/modified/deleted files, sends Slack/email alerts, downloads files, and updates index.
- `file_analyzer.py`: extracts content/metadata from CSV, Excel, PDF, DOCX, PPTX, and text files.
- `metadata_indexer.py`: SQLite metadata store plus sentence-transformers embeddings.
- `semantic_search.py`: natural-language search with metadata filters and conversation memory for follow-up queries.
- `analytics_engine.py`: pandas analysis, trend/comparison charts, insights, and Excel export.
- `ai_router.py`: provider switch for Anthropic Claude, OpenAI, Groq, Gemini, HuggingFace, plus Amazon/web comparison hooks.
- `streamlit_app.py`: end-to-end dashboard for sync, search, summaries, analytics, basket workflows, renaming, and downloads.
- `config.py`: environment-based configuration and path setup.

## Project Structure

```text
drive-ai-assistant/
|-- config.py
|-- monitor_drive.py
|-- file_analyzer.py
|-- metadata_indexer.py
|-- semantic_search.py
|-- analytics_engine.py
|-- ai_router.py
|-- streamlit_app.py
|-- requirements.txt
|-- README.md
|-- data/
    |-- cache/
    |-- exports/
    |-- drive_assistant.db
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create environment variables:

```bash
GOOGLE_DRIVE_AUTH_MODE=service_account
GOOGLE_SERVICE_ACCOUNT_FILE=credentials.json
GOOGLE_SERVICE_ACCOUNT_JSON=""
GOOGLE_OAUTH_CLIENT_FILE=credentials.json
GOOGLE_OAUTH_TOKEN_FILE=token.json
DRIVE_FOLDER_ID=your_folder_id
GOOGLE_DRIVE_FOLDER_ID=your_folder_id  # optional alias
GOOGLE_SHARED_DRIVE_ID=your_shared_drive_id_optional
MONITOR_INTERVAL_SECONDS=300

SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_password_or_app_password
EMAIL_FROM=your_email
EMAIL_TO=target_email

AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GROQ_API_KEY=...
GEMINI_API_KEY=...
HUGGINGFACE_API_KEY=...
AMAZON_MARKETPLACE=amazon.in
BUSINESS_ROLE=leadership
EMBEDDING_BACKEND=hashing
```

If you mainly use Claude, set:

```bash
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_claude_api_key
```

## Google Drive Access

1. Create a Google Cloud project and enable Drive API.
2. Choose one auth mode:

### Option A: Service account (recommended for backend jobs)

1. Create a Service Account and download JSON key.
2. Put the file in project root as `credentials.json` (or set `GOOGLE_SERVICE_ACCOUNT_FILE` to full path).
3. Share your target Drive folder or the whole Shared Drive with the service account email.
4. If you want the app to rename files from Streamlit, give the service account `Editor` or `Content manager` access. `Viewer` is enough for read/search/download only.
5. Set `GOOGLE_DRIVE_AUTH_MODE=service_account`.

### Shared Drive scan modes

Folder mode:
- Set `DRIVE_FOLDER_ID` to scan one folder.
- `GOOGLE_SHARED_DRIVE_ID` is optional helper for some shared drive setups.

Whole Shared Drive mode:
- Leave `DRIVE_FOLDER_ID` empty.
- Set `GOOGLE_SHARED_DRIVE_ID` to the Shared Drive ID.
- The app will scan the entire Shared Drive.

### Option B: OAuth user login

1. Create OAuth Client ID (Desktop app) in Google Cloud.
2. Download client secret JSON and place it in project root as `credentials.json`.
3. Set `GOOGLE_DRIVE_AUTH_MODE=oauth`.
4. On first run, browser login opens and `token.json` will be created automatically.

### About "Google API key"

For this project you do not use a simple API key for Drive file access.  
You must use Service Account or OAuth credentials JSON because Drive listing/download requires authenticated user scopes.

## Data Source Behavior

At startup the app follows this order:

1. Check whether real Drive data is already indexed in SQLite.
2. If indexed real data exists: use it immediately.
3. Otherwise: load demo data fast so Streamlit can start cleanly.
4. Use `Run Monitor Once` from the sidebar to fetch fresh Google Drive data on demand.

UI shows one of:
- `Using Google Drive data`
- `Using demo data`

If a file has no Drive URL (demo/local), UI shows `Local/demo file (no Drive link)`.

## LLM Cost Control

The sidebar now includes:

- `Enable paid LLM`: turns Claude/OpenAI/Groq/Gemini/HuggingFace calls on or off
- `LLM provider`: lets you switch providers without changing code

When LLM mode is off:

- retrieval, folder/file search, rename preview, combine/export, dashboard metrics, downloads, and alerts still work
- intent routing uses rule-based logic
- summaries, briefs, and dataset commentary use deterministic fallback text

This is the recommended default mode if you want to keep API costs controlled.

## Fast Startup Embeddings

The app now defaults to:

```bash
EMBEDDING_BACKEND=hashing
```

This avoids large model downloads on Streamlit Cloud and keeps app startup fast enough for health checks.

You can still change this later if you want a heavier embedding stack, but `hashing` is the safest deployment choice.

## Business Workflows

The app now supports:

- folder-aware search such as `show folders available` and `show contents of folder finance`
- selected-item basket for combining files and building dashboards
- executive briefs and business alerts from selected tabular datasets
- filename cleanup preview via prompts like `clean naming in folder finance`
- Drive rename apply via prompts like `apply rename in folder finance`
- Amazon marketplace comparison via prompts like `compare this dataset with Amazon`
- direct browser download buttons for any indexed file with a local cached copy

## Run

### Start monitoring service

```bash
python monitor_drive.py
```

### Run Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

### Run integration test flow

```bash
python test_flow.py
```

This flow performs:
1. fetch files from Drive folder
2. index new/updated files into SQLite
3. run sample query: `Show files uploaded by Amber`

## Natural Language Search Examples

- `Show files uploaded by Amber`
- `Find datasets related to revenue and profit`
- `Show sales datasets for 2025`
- `Find marketing documents`
- Follow-up: `From these, which contain revenue data?`

## Streamlit Cloud Deployment (GitHub)

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, click `New app`.
3. Select repo and set entry point: `streamlit_app.py`.
4. Add all required secrets in Streamlit app settings.
5. Deploy.
6. Trigger initial sync using `Run Monitor Once` from sidebar.

### Recommended Streamlit secrets for service account deployment

Do not upload `credentials.json` to GitHub.  
Instead, paste the JSON into Streamlit secrets like this.

Important: keep `AI_PROVIDER` and `OPENAI_API_KEY` above the `[GOOGLE_SERVICE_ACCOUNT_JSON]` section so they stay top-level in TOML.

```toml
GOOGLE_DRIVE_AUTH_MODE = "service_account"
AI_PROVIDER = "anthropic"
ANTHROPIC_API_KEY = "your_claude_key"
OPENAI_API_KEY = "your_openai_key_if_needed"
DRIVE_FOLDER_ID = "your_folder_id"
GOOGLE_SHARED_DRIVE_ID = ""

[GOOGLE_SERVICE_ACCOUNT_JSON]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = """-----BEGIN PRIVATE KEY-----
...
-----END PRIVATE KEY-----
"""
client_email = "..."
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"
```

If you want to scan just one folder inside the Shared Drive:

```toml
DRIVE_FOLDER_ID = "your_folder_id"
GOOGLE_SHARED_DRIVE_ID = "your_shared_drive_id"
```

## Notes

- SQLite database is created automatically at `data/drive_assistant.db`.
- Downloaded Drive files are cached under `data/cache/`.
- Exported analytics reports are generated under `data/exports/`.
- OAuth token file is saved at `token.json` (or `GOOGLE_OAUTH_TOKEN_FILE`).
- Amazon comparison uses best-effort marketplace parsing and may fail if Amazon blocks automated requests; the app falls back gracefully.
- If you use Claude, the simplest deployment path is `AI_PROVIDER=anthropic` plus `ANTHROPIC_API_KEY`, then leave `Enable paid LLM` off until you specifically need narrative summaries.
