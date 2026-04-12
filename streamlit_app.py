from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
from PIL import Image

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

from ai_router import AIRouter
from analytics_engine import AnalyticsEngine
from config import DASHBOARD_PRESETS_PATH, EXPORT_DIR, settings
from metadata_indexer import MetadataIndexer
from monitor_drive import DriveMonitor
from semantic_search import SemanticSearchEngine
from styles import inject_styles


st.set_page_config(page_title="Drive AI Assistant", layout="wide")
st.title("Drive AI Assistant")
st.caption("Conversational Drive intelligence with memory, analytics, and exports")
inject_styles()


@st.cache_resource
def get_services():
    indexer = MetadataIndexer()
    search = SemanticSearchEngine(indexer)
    analytics = AnalyticsEngine()
    return indexer, search, analytics


@st.cache_data(ttl=300, show_spinner=False)
def _cached_load_dataset(local_path: str, max_rows: int = 50_000) -> pd.DataFrame:
    """Cache loaded DataFrames for 5 minutes keyed by file path.

    Using a module-level cached function (not a method) so Streamlit can
    serialize the cache key without touching the AnalyticsEngine instance.
    Re-uses the singleton analytics engine from get_services().
    """
    _, _, analytics = get_services()
    return analytics.load_dataset(local_path, max_rows=max_rows)


def init_session() -> None:
    """Initialise all session state keys with safe defaults.

    Session state schema (all keys documented here for future maintainers):
      chat_history        List[Dict]    — chat messages {role, content, ...}
      chat_turn_counter   int           — increments per user turn; used for widget key namespacing
      last_results        List[Dict]    — file rows from the most recent search
      last_selected_files List[str]     — file IDs chosen for the active analysis context
      selected_items      List[str]     — persistent basket of file IDs across turns
      latest_results      List[Dict]    — alias kept for analytics tab compatibility
      latest_compare_df   DataFrame|None— cross-file comparison table from last compare action
      data_source         str           — "google_drive" | "demo" | "unknown"
      data_source_detail  str           — human-readable explanation of data source
      data_bootstrap_done bool          — prevents repeated bootstrap on every rerun
      rename_preview      List[Dict]    — pending rename plan rows
      executive_brief     Dict|None     — last generated executive brief payload
      latest_alerts       List[Dict]    — last generated business alert rows
      business_role       str           — active role (leadership/sales/finance/marketing/operations)
      llm_enabled         bool          — sidebar toggle for paid LLM usage
      llm_provider        str           — active LLM provider name
      explorer_folder_id  str           — folder ID open in Folder Explorer tab
      selected_preset_name str          — name of the active dashboard preset
      dataset_selection   List[str]     — multiselect widget value for dataset picker
      last_sync_time      datetime|None — UTC timestamp of last Drive sync (enforces 5-min TTL)
    """
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("chat_turn_counter", 0)
    st.session_state.setdefault("last_results", [])
    st.session_state.setdefault("last_selected_files", [])
    st.session_state.setdefault("selected_items", [])
    st.session_state.setdefault("latest_results", [])
    st.session_state.setdefault("latest_compare_df", None)
    st.session_state.setdefault("data_source", "unknown")
    st.session_state.setdefault("data_source_detail", "")
    st.session_state.setdefault("data_bootstrap_done", False)
    st.session_state.setdefault("rename_preview", [])
    st.session_state.setdefault("executive_brief", None)
    st.session_state.setdefault("latest_alerts", [])
    st.session_state.setdefault("business_role", settings.business_role)
    st.session_state.setdefault("llm_enabled", False)
    st.session_state.setdefault("explorer_folder_id", "")
    st.session_state.setdefault("selected_preset_name", "")
    st.session_state.setdefault("dataset_selection", [])
    st.session_state.setdefault("last_sync_time", None)
    st.session_state.setdefault("llm_call_count", 0)
    st.session_state.setdefault("llm_token_estimate", 0)
    default_provider = settings.ai_provider
    if default_provider == "openai" and not settings.openai_api_key and settings.anthropic_api_key:
        default_provider = "anthropic"
    st.session_state.setdefault("llm_provider", default_provider)


def _ensure_demo_data(indexer: MetadataIndexer) -> int:
    repair = getattr(indexer, "ensure_dummy_data_ready", None)
    if callable(repair):
        return repair()
    seed = getattr(indexer, "seed_dummy_data_if_empty", None)
    if callable(seed):
        result = seed()
        if isinstance(result, bool):
            return 3 if result else 0
        return int(result or 0)
    return 0


def _reset_chat_and_context() -> None:
    st.session_state["chat_history"] = []
    st.session_state["chat_turn_counter"] = 0
    st.session_state["last_results"] = []
    st.session_state["last_selected_files"] = []
    st.session_state["selected_items"] = []
    st.session_state["latest_results"] = []
    st.session_state["latest_compare_df"] = None
    st.session_state["rename_preview"] = []
    st.session_state["executive_brief"] = None
    st.session_state["latest_alerts"] = []
    st.session_state["dataset_selection"] = []


def _load_dashboard_presets() -> List[Dict]:
    if not DASHBOARD_PRESETS_PATH.exists():
        return []
    try:
        data = json.loads(DASHBOARD_PRESETS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _sync_dataset_selection_from_widget() -> None:
    selected = st.session_state.get("dataset_selection", [])
    if not selected:
        return
    file_ids = [item.split("|")[0].strip() for item in selected if "|" in item]
    if not file_ids:
        return
    st.session_state["last_selected_files"] = file_ids
    st.session_state["selected_items"] = list(dict.fromkeys(st.session_state["selected_items"] + file_ids))


def _save_dashboard_presets(presets: List[Dict]) -> None:
    DASHBOARD_PRESETS_PATH.write_text(json.dumps(presets, indent=2), encoding="utf-8")


def _upsert_dashboard_preset(preset: Dict) -> None:
    presets = _load_dashboard_presets()
    presets = [item for item in presets if item.get("name") != preset.get("name")]
    presets.append(preset)
    _save_dashboard_presets(sorted(presets, key=lambda item: item.get("name", "").lower()))


def _delete_dashboard_preset(name: str) -> None:
    presets = [item for item in _load_dashboard_presets() if item.get("name") != name]
    _save_dashboard_presets(presets)


def _apply_preset_to_state(preset: Dict) -> None:
    for key, value in preset.get("state", {}).items():
        st.session_state[key] = value
    dataset_ids = preset.get("dataset_ids", [])
    if dataset_ids:
        st.session_state["last_selected_files"] = dataset_ids
        st.session_state["selected_items"] = list(dict.fromkeys(st.session_state["selected_items"] + dataset_ids))


def _add_selected_item(file_id: str) -> None:
    basket = st.session_state["selected_items"]
    if file_id not in basket:
        basket.append(file_id)
    st.session_state["selected_items"] = basket


def _remove_selected_item(file_id: str) -> None:
    st.session_state["selected_items"] = [item for item in st.session_state["selected_items"] if item != file_id]


def _extract_target_name(query: str) -> str | None:
    patterns = [
        r"(?:download|analyze|rename|open|compare)\s+(?:file|folder)?\s*([A-Za-z0-9 _./()-]+)",
        r"(?:inside|under|in)\s+folder\s+([A-Za-z0-9 _./()-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,!?:;")
    return None


def _resolve_context_items(indexer: MetadataIndexer, query: str, item_kind: str | None = None) -> List[Dict]:
    selected_ids = st.session_state["selected_items"] or st.session_state["last_selected_files"]
    items = []
    for file_id in selected_ids:
        info = indexer.get_file(file_id)
        if not info:
            continue
        if item_kind and info.get("item_kind") != item_kind:
            continue
        items.append(info)
    if items:
        return items

    target_name = _extract_target_name(query)
    if target_name:
        matches = indexer.search_by_name(target_name, item_kind=item_kind, limit=12)
        if matches:
            return matches

    if st.session_state["last_results"]:
        return [
            item
            for item in st.session_state["last_results"]
            if not item_kind or item.get("item_kind", "file") == item_kind
        ]
    return indexer.list_items(item_kind)[:12]


def bootstrap_data_source(indexer: MetadataIndexer) -> None:
    if st.session_state.get("data_bootstrap_done"):
        return

    real_count = indexer.count_real_files()
    if real_count > 0:
        indexer.remove_dummy_data()
        source = "google_drive"
        detail = f"Using Google Drive data ({real_count} files already indexed)."
    else:
        seeded = _ensure_demo_data(indexer)
        source = "demo"
        detail = (
            "Startup uses demo data by default. Click `Run Monitor Once` to sync Google Drive."
            if seeded > 0
            else "No indexed data yet. Click `Run Monitor Once` to sync Google Drive."
        )

    st.session_state["data_source"] = source
    st.session_state["data_source_detail"] = detail
    st.session_state["data_bootstrap_done"] = True


def _result_card(r: dict, key_prefix: str = "live", interactive: bool = True) -> None:
    keywords = ", ".join(json.loads(r.get("keywords", "[]")))
    link = r.get("file_link", "")
    item_kind = r.get("item_kind", "file")
    if link:
        link_line = f"- [Open file]({link})"
    else:
        link_line = "- Local/demo file (no Drive link)"
    st.markdown(
        f"**{r['file_name']}**  \n"
        f"- Kind: `{item_kind}`  \n"
        f"- Path: `{r.get('path_text', r['file_name'])}`  \n"
        f"- Uploader: `{r['uploader_name']}`  \n"
        f"- Type: `{r['file_type']}`  \n"
        f"- Modified: `{r['modified_time']}`  \n"
        f"- Topic: `{r['topic']}`  \n"
        f"- Keywords: `{keywords}`  \n"
        f"- Similarity score: `{r.get('score', '-')}`  \n"
        f"{link_line}"
    )
    summary_text = (r.get("summary") or "").strip()
    if len(summary_text) > 220:
        summary_text = summary_text[:220].rstrip() + "..."
    if summary_text:
        st.write(summary_text)
    if not interactive:
        return
    col1, col2 = st.columns(2)
    with col1:
        if r["file_id"] in st.session_state["selected_items"]:
            if st.button(f"Remove from basket: {r['file_name']}", key=f"{key_prefix}_remove_{r['file_id']}"):
                _remove_selected_item(r["file_id"])
        else:
            if st.button(f"Add to basket: {r['file_name']}", key=f"{key_prefix}_add_{r['file_id']}"):
                _add_selected_item(r["file_id"])
    with col2:
        if item_kind == "file" and r.get("local_path") and Path(r["local_path"]).exists():
            with open(r["local_path"], "rb") as file_handle:
                st.download_button(
                    f"Download {r['file_name']}",
                    data=file_handle.read(),
                    file_name=r["file_name"],
                    key=f"{key_prefix}_download_chat_{r['file_id']}",
                )


def _load_context_datasets(
    indexer: MetadataIndexer,
    analytics: AnalyticsEngine,
    query: str = "",
    prefer_selected: bool = False,
) -> List[Dict]:
    _sync_dataset_selection_from_widget()
    query_lower = query.lower()
    use_basket = prefer_selected or any(token in query_lower for token in ["selected", "basket", "picked"])
    if st.session_state["last_selected_files"]:
        file_ids = st.session_state["last_selected_files"]
    elif use_basket and st.session_state["selected_items"]:
        file_ids = st.session_state["selected_items"]
    else:
        file_ids = [r["file_id"] for r in st.session_state["last_results"]]
        if not file_ids and st.session_state["selected_items"]:
            file_ids = st.session_state["selected_items"]
    datasets = []
    for fid in file_ids:
        info = indexer.get_file(fid)
        if not info or info.get("item_kind") != "file":
            continue
        suffix = Path(info.get("local_path", "")).suffix.lower()
        if suffix not in [".csv", ".xlsx", ".xls"]:
            continue
        df = _cached_load_dataset(info["local_path"])
        if not df.empty:
            datasets.append({"file_name": info["file_name"], "df": df, "info": info})
    return datasets


def _show_chat_history() -> None:
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("for_query"):
                st.caption(f"Response to: {msg['for_query']}")
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("results"):
                with st.expander(f"Results for this question ({len(msg['results'])})", expanded=False):
                    for idx, result in enumerate(msg["results"][:8], start=1):
                        st.markdown(f"**{idx}. {result['file_name']}**  \nKind: `{result.get('item_kind', 'file')}`  \nUploader: `{result.get('uploader_name', '-')}`")
            if msg["role"] == "assistant" and msg.get("compare_rows"):
                with st.expander("Comparison output", expanded=False):
                    st.dataframe(pd.DataFrame(msg["compare_rows"]), use_container_width=True)
            if msg["role"] == "assistant" and msg.get("analysis_rows"):
                with st.expander("Current dataset analysis", expanded=False):
                    st.dataframe(pd.DataFrame(msg["analysis_rows"]), use_container_width=True)
            if msg["role"] == "assistant" and msg.get("rename_preview"):
                with st.expander("Rename preview", expanded=False):
                    st.dataframe(pd.DataFrame(msg["rename_preview"]), use_container_width=True)
            if msg["role"] == "assistant" and msg.get("alerts_rows"):
                with st.expander("Alerts", expanded=False):
                    st.dataframe(pd.DataFrame(msg["alerts_rows"]), use_container_width=True)
            if msg["role"] == "assistant" and msg.get("market_rows"):
                with st.expander("Market comparison", expanded=False):
                    st.dataframe(pd.DataFrame(msg["market_rows"]), use_container_width=True)


def _summarize_results_rule_based(results: List[Dict]) -> str:
    if not results:
        return "No Drive items matched the current request."
    topics = {}
    for item in results:
        topic = item.get("topic", "general")
        topics[topic] = topics.get(topic, 0) + 1
    top_topics = ", ".join([f"{topic} ({count})" for topic, count in sorted(topics.items(), key=lambda pair: pair[1], reverse=True)[:4]])
    folders = len([item for item in results if item.get("item_kind") == "folder"])
    files = len(results) - folders
    return f"Current result set contains {files} files and {folders} folders. Strongest themes: {top_topics}."


def _summarize_brief_rule_based(brief: Dict) -> str:
    headlines = brief.get("headlines", [])
    alerts = brief.get("alerts", [])
    parts = []
    if headlines:
        parts.append("Key headlines: " + " | ".join(headlines[:3]))
    if alerts:
        parts.append("Key risks: " + " | ".join([alert["message"] for alert in alerts[:3]]))
    if not parts:
        return "Executive brief ready. Current selection has structured data but no major risks or trends were detected."
    return "\n\n".join(parts)


def _summarize_dataset_rule_based(file_name: str, metrics: Dict, anomalies: Dict, extra_insights: List[str]) -> str:
    lines = [
        f"{file_name} has {metrics.get('rows', 0)} rows and {metrics.get('columns', 0)} columns.",
    ]
    if "default_metric_column" in metrics:
        lines.append(f"Primary metric appears to be `{metrics['default_metric_column']}`.")
    if "sum" in metrics and "average" in metrics:
        lines.append(f"Total is {metrics['sum']:,.2f} and average is {metrics['average']:,.2f}.")
    lines.extend(extra_insights[:2])
    if anomalies.get("count", 0):
        lines.append(f"{anomalies['count']} unusual observations need review.")
    return " ".join(lines)


def _summarize_market_compare_rule_based(compare_df: pd.DataFrame) -> str:
    if compare_df.empty:
        return "No market comparison could be produced from the current dataset."
    cheapest = compare_df.sort_values("market_price").iloc[0]
    highest_gap = compare_df.iloc[compare_df["price_gap"].abs().idxmax()]
    return (
        f"Cheapest matching Amazon listing is `{cheapest['market_item']}` at {cheapest['market_price']:,.2f}. "
        f"The largest gap versus your internal average is `{highest_gap['market_item']}` with a gap of {highest_gap['price_gap']:,.2f}."
    )


def _looks_like_current_dataset_question(query: str) -> bool:
    query_lower = query.lower()
    negative_triggers = [
        "show files",
        "show folders",
        "list files",
        "list folders",
        "uploaded by",
        "find file",
        "find folder",
    ]
    if any(token in query_lower for token in negative_triggers):
        return False
    triggers = [
        "current file",
        "this file",
        "current dataset",
        "selected file",
        "selected dataset",
        "in this",
        "in current",
        "asin wise",
        "group by",
        "top ",
        "highest ",
        "lowest ",
        "trend",
        "average",
        "total",
        "sum",
        "sorted by",
        "wise",
        "group by",
    ]
    return any(token in query_lower for token in triggers)


def _should_answer_from_current_dataset(query: str) -> bool:
    _sync_dataset_selection_from_widget()
    if not st.session_state["last_selected_files"]:
        return False
    if _looks_like_current_dataset_question(query):
        return True
    query_lower = query.lower()
    analytic_terms = [
        "analyze",
        "analyse",
        "trend",
        "top",
        "highest",
        "lowest",
        "average",
        "sum",
        "total",
        "group by",
        "wise",
        "sorted by",
    ]
    retrieval_terms = ["show files", "show folders", "list files", "list folders", "uploaded by"]
    return any(term in query_lower for term in analytic_terms) and not any(term in query_lower for term in retrieval_terms)


def _render_column_guide(profile: Dict) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Numeric")
        for item in profile["numeric_columns"][:12]:
            st.write(f"- {item}")
    with col2:
        st.caption("Categorical")
        for item in profile["categorical_columns"][:12]:
            st.write(f"- {item}")
    with col3:
        st.caption("Time / Suggested")
        for item in profile["datetime_columns"][:8]:
            st.write(f"- {item}")
        if profile["metric_suggestions"]:
            st.caption("Suggested metrics")
            for item in profile["metric_suggestions"][:8]:
                st.write(f"- {item}")


def _scroll_chat_to_bottom() -> None:
    st.components.v1.html(
        """
        <script>
        const scrollTarget = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
        if (scrollTarget) {
            window.parent.requestAnimationFrame(() => {
                window.parent.scrollTo({ top: scrollTarget.scrollHeight, behavior: "smooth" });
            });
        }
        </script>
        """,
        height=0,
    )


def _render_kpi_cards(
    kpis: List[Dict],
    key_prefix: str,
    definitions: Dict[str, str] | None = None,
    analytics: AnalyticsEngine | None = None,
) -> None:
    if not kpis:
        st.caption("No numeric KPI metrics available.")
        return
    definitions = definitions or {}
    cols = st.columns(min(4, len(kpis)))
    for idx, kpi in enumerate(kpis[:4]):
        with cols[idx]:
            formatted_sum = analytics.format_metric_value(kpi["metric"], kpi["sum"]) if analytics else f"{kpi['sum']:,.2f}"
            formatted_avg = analytics.format_metric_value(kpi["metric"], kpi["average"]) if analytics else f"{kpi['average']:,.2f}"
            formatted_median = analytics.format_metric_value(kpi["metric"], kpi["median"]) if analytics else f"{kpi['median']:,.2f}"
            help_text = f"Avg: {formatted_avg} | Median: {formatted_median}"
            if kpi["metric"] in definitions:
                help_text = f"{definitions[kpi['metric']]}\n\n{help_text}"
            st.metric(analytics.humanize_label(kpi["metric"]) if analytics else kpi["metric"], formatted_sum, help=help_text)


def _render_benchmark_cards(cards: List[Dict]) -> None:
    if not cards:
        st.caption("No benchmark cards available for the current dataset.")
        return
    chunks = []
    for card in cards[:4]:
        label_lower = card["label"].lower()
        card_class = "neutral"
        if any(token in label_lower for token in ["best", "fastest", "highest"]):
            card_class = "good"
        elif any(token in label_lower for token in ["weakest", "lowest"]):
            card_class = "warn"
        chunks.append(
            f"""
            <div class="benchmark-card {card_class}">
                <div class="benchmark-label">{card['label']}</div>
                <div class="benchmark-value">{card['value']}</div>
                <div class="benchmark-delta">{card['formatted']}</div>
                <div class="benchmark-detail">{card['detail']}</div>
            </div>
            """
        )
    st.markdown(f"<div class='benchmark-grid'>{''.join(chunks)}</div>", unsafe_allow_html=True)


def _render_chart_reasons(chart_reasons: List[Dict]) -> None:
    if not chart_reasons:
        st.caption("No chart-selection notes available.")
        return
    for item in chart_reasons:
        st.markdown(f"- **{item['chart_type'].title()}**: {item['reason']}")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")


def _render_anchor(anchor_id: str) -> None:
    st.markdown(f"<div id='{anchor_id}'></div>", unsafe_allow_html=True)


def _render_dataset_mini_nav(dataset_id: str) -> None:
    sections = [
        ("Summary", f"{dataset_id}-summary"),
        ("Metric guide", f"{dataset_id}-metric-guide"),
        ("Benchmarks", f"{dataset_id}-benchmarks"),
        ("Full table", f"{dataset_id}-full-table"),
        ("KPIs", f"{dataset_id}-kpis"),
        ("Best dashboard", f"{dataset_id}-best-dashboard"),
        ("Builder", f"{dataset_id}-builder"),
        ("Anomalies", f"{dataset_id}-anomalies"),
        ("Exports", f"{dataset_id}-exports"),
    ]
    links = "".join([f"<a href='#{anchor}'>{label}</a>" for label, anchor in sections])
    st.markdown(
        f"""
        <details class="dataset-mini-nav" open>
            <summary>Jump within this dataset</summary>
            <div>{links}</div>
        </details>
        """,
        unsafe_allow_html=True,
    )


def _render_dataset_toolbar(
    file_name: str,
    dataset_type: str,
    row_count: int,
    column_count: int,
    primary_metric: str | None,
    time_dim: str | None,
    category_dim: str | None,
) -> None:
    metric_text = primary_metric or "No primary metric detected"
    time_text = time_dim or "No time grain detected"
    category_text = category_dim or "No category grain detected"
    st.markdown(
        f"""
        <div class="dataset-toolbar">
            <div class="dataset-toolbar-title">{file_name}</div>
            <div class="dataset-toolbar-meta">
                <strong>Type:</strong> {dataset_type.title()} |
                <strong>Rows:</strong> {row_count:,} |
                <strong>Columns:</strong> {column_count} |
                <strong>Primary metric:</strong> {metric_text} |
                <strong>Time grain:</strong> {time_text} |
                <strong>Category grain:</strong> {category_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chart_grid(figures: List) -> None:
    if not figures:
        return
    cols = st.columns(2)
    for idx, fig in enumerate(figures):
        with cols[idx % 2]:
            st.plotly_chart(fig, use_container_width=True)


def _figure_png_bytes(fig):
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


def _dashboard_pdf_bytes(title: str, panels: List[Dict]) -> bytes | None:
    if not REPORTLAB_AVAILABLE:
        return None
    png_images = []
    for panel in panels:
        png_bytes = _figure_png_bytes(panel["figure"])
        if png_bytes:
            png_images.append((panel["chart_type"], png_bytes))
    if not png_images:
        return None

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    for idx, (chart_type, png_bytes) in enumerate(png_images):
        image = Image.open(io.BytesIO(png_bytes))
        image_width, image_height = image.size
        max_width = width - 72
        max_height = height - 140
        scale = min(max_width / image_width, max_height / image_height)
        draw_width = image_width * scale
        draw_height = image_height * scale
        x = (width - draw_width) / 2
        y = height - draw_height - 80
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(36, height - 40, title)
        pdf.setFont("Helvetica", 11)
        pdf.drawString(36, height - 58, f"Chart {idx + 1}: {chart_type.title()}")
        pdf.drawImage(ImageReader(io.BytesIO(png_bytes)), x, y, width=draw_width, height=draw_height)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _render_chart_panels(panels: List[Dict], dataset_id: str, section_key: str) -> None:
    if not panels:
        return
    cols = st.columns(2)
    for idx, panel in enumerate(panels):
        with cols[idx % 2]:
            st.plotly_chart(panel["figure"], use_container_width=True)
            btn_cols = st.columns(2)
            png_bytes = _figure_png_bytes(panel["figure"])
            with btn_cols[0]:
                if png_bytes:
                    st.download_button(
                        f"PNG: {panel['chart_type'].title()}",
                        data=png_bytes,
                        file_name=f"{dataset_id}_{section_key}_{idx + 1}_{panel['chart_type']}.png",
                        mime="image/png",
                        key=f"{section_key}_png_{dataset_id}_{idx}",
                    )
                else:
                    st.caption("PNG export unavailable")
            with btn_cols[1]:
                source_df = panel.get("data", pd.DataFrame())
                if isinstance(source_df, pd.DataFrame) and not source_df.empty:
                    st.download_button(
                        f"CSV: {panel['chart_type'].title()}",
                        data=source_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{dataset_id}_{section_key}_{idx + 1}_{panel['chart_type']}.csv",
                        mime="text/csv",
                        key=f"{section_key}_csv_{dataset_id}_{idx}",
                    )


def _safe_download_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "_", name).strip()
    return cleaned or fallback


def _build_renamed_zip(file_items: List[Dict], rename_map: Dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in file_items:
            local_path = item.get("local_path", "")
            if not local_path or not Path(local_path).exists():
                continue
            target_name = _safe_download_name(rename_map.get(item["file_id"], item["file_name"]), item["file_name"])
            zf.write(local_path, arcname=target_name)
    buffer.seek(0)
    return buffer.getvalue()


def _render_file_actions(indexer: MetadataIndexer, analytics: AnalyticsEngine) -> None:
    st.subheader("File Actions")
    all_files = indexer.list_items("file")
    if not all_files:
        st.caption("No indexed files available yet.")
        return

    with st.expander("Rename, Download, Analyze, or Merge Files", expanded=False):
        file_options = [f"{item['file_id']} | {item['file_name']}" for item in all_files]
        selected_files = st.multiselect("Select one or more files", options=file_options, key="file_actions_selection")
        selected_items = []
        selected_ids = []
        for option in selected_files:
            file_id = option.split("|")[0].strip()
            info = indexer.get_file(file_id)
            if info:
                selected_items.append(info)
                selected_ids.append(file_id)

        rename_tab, analysis_tab = st.tabs(["Rename & Download", "Multi-file Analysis"])

        with rename_tab:
            if not selected_items:
                st.caption("Select files above to rename or download them.")
            else:
                monitor = DriveMonitor()
                rename_map: Dict[str, str] = {}
                for item in selected_items:
                    default_name = monitor.suggest_clean_name(item["file_name"], item.get("item_kind", "file"))
                    rename_map[item["file_id"]] = st.text_input(
                        f"New name for {item['file_name']}",
                        value=default_name,
                        key=f"rename_input_{item['file_id']}",
                    )

                if len(selected_items) == 1:
                    item = selected_items[0]
                    local_path = item.get("local_path", "")
                    if local_path and Path(local_path).exists():
                        with open(local_path, "rb") as handle:
                            st.download_button(
                                "Download renamed file",
                                data=handle.read(),
                                file_name=_safe_download_name(rename_map[item["file_id"]], item["file_name"]),
                                key=f"download_renamed_single_{item['file_id']}",
                            )
                else:
                    zip_bytes = _build_renamed_zip(selected_items, rename_map)
                    st.download_button(
                        "Download renamed files as ZIP",
                        data=zip_bytes,
                        file_name="renamed_files.zip",
                        mime="application/zip",
                        key="download_renamed_zip",
                    )

                if st.session_state["data_source"] == "google_drive":
                    if st.button("Apply these names in Google Drive", key="apply_custom_drive_renames"):
                        monitor = DriveMonitor()
                        rename_plan = []
                        for item in selected_items:
                            new_name = rename_map.get(item["file_id"], item["file_name"]).strip()
                            if new_name and new_name != item["file_name"]:
                                rename_plan.append(
                                    {
                                        "file_id": item["file_id"],
                                        "item_kind": item.get("item_kind", "file"),
                                        "old_name": item["file_name"],
                                        "new_name": new_name,
                                        "will_change": True,
                                    }
                                )
                        if not rename_plan:
                            st.info("No actual file-name changes were entered.")
                        else:
                            applied = monitor.apply_rename_plan(rename_plan)
                            if applied:
                                st.success(f"Applied {len(applied)} Drive renames. Run sync once to refresh the index.")
                            else:
                                st.warning("No Drive renames were applied. This usually means the service account lacks edit access.")

        with analysis_tab:
            tabular_items = [
                item for item in selected_items if Path(item.get("local_path", "")).suffix.lower() in [".csv", ".xlsx", ".xls"]
            ]
            if not tabular_items:
                st.caption("Select one or more CSV/Excel files above to analyze or merge them.")
            else:
                if st.button("Use selected files for analysis", key="load_selected_file_actions_context"):
                    st.session_state["last_selected_files"] = selected_ids
                    st.session_state["selected_items"] = list(dict.fromkeys(st.session_state["selected_items"] + selected_ids))
                    st.success("Selected files loaded into the current analysis context below.")
                    st.rerun()

                if len(tabular_items) >= 2:
                    merge_mode = st.selectbox(
                        "Merge mode",
                        options=["Append rows", "Only common columns"],
                        key="merge_mode_select",
                    )
                    merge_datasets = []
                    for item in tabular_items:
                        df = _cached_load_dataset(item["local_path"])
                        if not df.empty:
                            merge_datasets.append({"file_name": item["file_name"], "df": df})
                    if merge_datasets:
                        merged_df = analytics.combine_datasets(merge_datasets)
                        if merge_mode == "Only common columns" and not merged_df.empty:
                            common_cols = set(merge_datasets[0]["df"].columns)
                            for dataset in merge_datasets[1:]:
                                common_cols &= set(dataset["df"].columns)
                            if common_cols:
                                ordered_cols = [col for col in merged_df.columns if col in common_cols or col == "source_file"]
                                merged_df = merged_df[ordered_cols]
                        st.caption(f"Merged preview across {len(merge_datasets)} files")
                        st.dataframe(merged_df.head(50), use_container_width=True)
                        st.download_button(
                            "Download merged CSV",
                            data=merged_df.to_csv(index=False).encode("utf-8"),
                            file_name="merged_analysis.csv",
                            mime="text/csv",
                            key="download_merged_csv",
                        )
                        merged_xlsx = io.BytesIO()
                        with pd.ExcelWriter(merged_xlsx, engine="openpyxl") as writer:
                            merged_df.to_excel(writer, index=False, sheet_name="MergedData")
                        st.download_button(
                            "Download merged Excel",
                            data=merged_xlsx.getvalue(),
                            file_name="merged_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_merged_xlsx",
                        )


def _save_builder_preset_ui(
    preset_scope: str,
    dataset_ids: List[str],
    state_keys: List[str],
    label: str,
) -> None:
    preset_name = st.text_input("Preset name", key=f"{preset_scope}_preset_name", placeholder=f"{label} preset")
    if st.button(f"Save preset for {label}", key=f"{preset_scope}_save_preset"):
        if not preset_name.strip():
            st.warning("Enter a preset name before saving.")
        else:
            preset = {
                "name": preset_name.strip(),
                "scope": preset_scope,
                "dataset_ids": dataset_ids,
                "state": {key: st.session_state.get(key) for key in state_keys},
            }
            _upsert_dashboard_preset(preset)
            st.success(f"Saved preset `{preset_name.strip()}`.")


def _render_dataset_dashboard(
    df: pd.DataFrame,
    analytics: AnalyticsEngine,
    file_label: str,
    key_prefix: str,
    dataset_id: str,
) -> None:
    profile = analytics.dataset_profile(df)
    dataset_type = analytics.infer_dataset_type(df)
    metric_definition_map = {item["metric"]: item["definition"] for item in analytics.metric_definitions(df)}
    st.caption(
        f"{file_label}: {profile['row_count']} rows, {profile['column_count']} columns, "
        f"{len(profile['numeric_columns'])} numeric fields, {len(profile['categorical_columns'])} grouping fields."
    )
    st.caption(f"Detected dataset type: `{dataset_type}`")
    if profile["metric_suggestions"]:
        st.caption(f"Suggested comparison metrics: {', '.join(profile['metric_suggestions'][:5])}")

    with st.expander(f"Dashboard Builder: {file_label}", expanded=True):
        use_all_metrics = st.checkbox("Use all numeric metrics", key=f"{key_prefix}_all_metrics")
        default_metrics = profile["metric_suggestions"] if use_all_metrics else profile["metric_suggestions"][: min(2, len(profile["metric_suggestions"]))]
        selected_metrics = st.multiselect(
            "Metrics to analyze",
            options=profile["numeric_columns"],
            default=[metric for metric in default_metrics if metric in profile["numeric_columns"]],
            key=f"{key_prefix}_metrics",
        )
        analysis_mode = st.radio(
            "Analysis mode",
            options=["dashboard", "multivariate", "univariate"],
            horizontal=True,
            key=f"{key_prefix}_mode",
        )
        agg = st.selectbox(
            "Aggregation",
            options=["sum", "mean", "median", "min", "max", "count"],
            index=0,
            key=f"{key_prefix}_agg",
        )
        category_col = st.selectbox(
            "Category dimension",
            options=[""] + profile["categorical_columns"],
            key=f"{key_prefix}_category",
        ) or None
        date_col = st.selectbox(
            "Time dimension",
            options=[""] + profile["datetime_columns"],
            key=f"{key_prefix}_date",
        ) or None
        chart_types = st.multiselect(
            "Chart types",
            options=analytics.available_chart_types(analysis_mode),
            default=analytics.available_chart_types(analysis_mode)[:3],
            key=f"{key_prefix}_charts",
        )
        _save_builder_preset_ui(
            preset_scope=f"dataset::{dataset_id}",
            dataset_ids=[dataset_id],
            state_keys=[
                f"{key_prefix}_all_metrics",
                f"{key_prefix}_metrics",
                f"{key_prefix}_mode",
                f"{key_prefix}_agg",
                f"{key_prefix}_category",
                f"{key_prefix}_date",
                f"{key_prefix}_charts",
            ],
            label=file_label,
        )

        if not selected_metrics:
            st.warning("Choose at least one numeric metric to build the dashboard.")
            return

        metric_kpis = analytics.metric_kpis(df, selected_metrics)
        _render_kpi_cards(metric_kpis, key_prefix=f"{key_prefix}_kpi", definitions=metric_definition_map, analytics=analytics)

        grouped_preview = analytics.aggregate_metrics(df, selected_metrics, group_col=date_col or category_col, agg=agg)
        if not grouped_preview.empty:
            preview_label = date_col or category_col or "overall dataset"
            st.caption(f"Preview table aggregated by `{preview_label}` using `{agg}`.")
            st.dataframe(grouped_preview.head(20), use_container_width=True)

        panels = analytics.dashboard_panels(
            df=df,
            metrics=selected_metrics,
            category_col=category_col,
            date_col=date_col,
            chart_types=chart_types,
            agg=agg,
        )
        with st.expander("Why these chart types work", expanded=False):
            _render_chart_reasons(
                [
                    {
                        "chart_type": chart_type,
                        "reason": analytics.explain_chart_choice(chart_type, selected_metrics, category_col, date_col, agg),
                    }
                    for chart_type in chart_types
                ]
            )
        if not panels:
            st.info("No compatible charts could be generated with the current selections. Try a different metric or dimension.")
        _render_chart_panels(panels, dataset_id=dataset_id, section_key=key_prefix)


def _render_metric_definitions(definitions: List[Dict]) -> None:
    if not definitions:
        st.caption("No suggested metric definitions available.")
        return
    for item in definitions:
        st.markdown(f"**{item['label']}**")
        st.caption(item["definition"])


def _render_grouped_anomaly_review(
    df: pd.DataFrame,
    analytics: AnalyticsEngine,
    default_bundle: Dict,
    file_id: str,
    primary_metric: str | None,
) -> None:
    if not primary_metric:
        st.caption("No primary numeric metric was detected, so anomaly review is unavailable.")
        return
    grain_options = ["Raw rows"] + default_bundle.get("grain_options", [])
    selected_grain = st.selectbox(
        "Anomaly grain",
        options=grain_options,
        key=f"anomaly_grain_{file_id}",
        help="Choose the business grain used to detect unusual values. Grouped anomalies are usually easier to trust than raw row outliers.",
    )
    if selected_grain == "Raw rows":
        anomalies = analytics.detect_anomalies(df, primary_metric)
        st.write(anomalies["method"] or "No anomaly method was applied.")
        if anomalies.get("bounds"):
            bounds = anomalies["bounds"]
            st.caption(
                f"Thresholds for `{primary_metric}`: lower {bounds['lower']:,.2f}, upper {bounds['upper']:,.2f}. "
                "Values outside this IQR range are flagged for review."
            )
        if anomalies["rows"]:
            st.dataframe(pd.DataFrame(anomalies["rows"]).head(20), use_container_width=True)
        else:
            st.caption("No raw-row anomalies were detected for the selected metric.")
        return

    grouped = analytics.detect_grouped_anomalies(df, primary_metric, selected_grain, agg="sum")
    st.write(grouped["method"] or "No grouped anomaly method was applied.")
    if grouped.get("bounds"):
        bounds = grouped["bounds"]
        st.caption(
            f"Grouped thresholds for `{primary_metric}` by `{selected_grain}`: lower {bounds['lower']:,.2f}, upper {bounds['upper']:,.2f}."
        )
    grouped_table = pd.DataFrame(grouped.get("grouped_table", []))
    if not grouped_table.empty:
        st.caption(f"Grouped view for anomaly detection by `{selected_grain}`")
        st.dataframe(grouped_table.head(30), use_container_width=True)
        grouped_chart = analytics.create_chart(grouped_table, "bar", [primary_metric], category_col=selected_grain, agg="sum")
        if grouped_chart:
            st.plotly_chart(grouped_chart, use_container_width=True)
    flagged_rows = pd.DataFrame(grouped["rows"])
    if not flagged_rows.empty:
        st.caption("Flagged grouped anomalies")
        st.dataframe(flagged_rows, use_container_width=True)
    else:
        st.caption(f"No grouped anomalies were detected by `{selected_grain}`.")


def _render_cross_file_dashboard(datasets: List[Dict], analytics: AnalyticsEngine, key_prefix: str) -> None:
    if len(datasets) < 2:
        return
    common_metrics = analytics.common_numeric_columns(datasets)
    st.subheader("Cross-file Dashboard")
    if not common_metrics:
        st.info("No common numeric metric names were found across the selected datasets, so cross-file comparison is limited.")
        return

    with st.expander("Compare selected datasets", expanded=True):
        compare_metrics = st.multiselect(
            "Metrics to compare across files",
            options=common_metrics,
            default=common_metrics[: min(2, len(common_metrics))],
            key=f"{key_prefix}_cross_metrics",
        )
        agg = st.selectbox(
            "Cross-file aggregation",
            options=["sum", "mean", "median", "min", "max", "count"],
            key=f"{key_prefix}_cross_agg",
        )
        chart_type = st.selectbox(
            "Cross-file chart type",
            options=["column", "bar", "line", "heatmap"],
            key=f"{key_prefix}_cross_chart",
        )
        _save_builder_preset_ui(
            preset_scope="cross_file",
            dataset_ids=[item["info"]["file_id"] for item in datasets if item.get("info")],
            state_keys=[
                f"{key_prefix}_cross_metrics",
                f"{key_prefix}_cross_agg",
                f"{key_prefix}_cross_chart",
            ],
            label="cross-file dashboard",
        )
        if not compare_metrics:
            st.warning("Choose at least one common metric.")
            return
        compare_df = analytics.compare_metrics_across_files(datasets, compare_metrics, agg=agg)
        if compare_df.empty:
            st.info("No comparable values were produced for the selected metrics.")
            return
        st.dataframe(compare_df, use_container_width=True)
        if chart_type == "heatmap":
            fig = analytics.create_chart(compare_df, "heatmap", metrics=compare_metrics)
        else:
            fig = analytics.create_chart(compare_df, chart_type, metrics=compare_metrics, category_col="file_name")
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def _render_drive_explorer(indexer: MetadataIndexer) -> None:
    st.subheader("Drive Explorer")
    folders = indexer.list_items("folder")
    if not folders:
        st.caption("No indexed folders available yet. Run Drive sync first.")
        return

    folder_options = {f"{folder.get('path_text') or folder['file_name']}": folder["file_id"] for folder in folders}
    labels = [""] + list(folder_options.keys())
    current_folder_id = st.session_state.get("explorer_folder_id", "")
    default_label = next((label for label, folder_id in folder_options.items() if folder_id == current_folder_id), "")
    selected_label = st.selectbox(
        "Browse folder",
        options=labels,
        index=labels.index(default_label) if default_label in labels else 0,
        key="explorer_folder_label",
    )
    selected_folder_id = folder_options.get(selected_label, "")
    st.session_state["explorer_folder_id"] = selected_folder_id

    if not selected_folder_id:
        st.caption("Choose a folder to browse indexed files and subfolders.")
        return

    overview = indexer.folder_overview(selected_folder_id)
    folder = overview["folder"]
    st.caption(f"Browsing `{folder.get('path_text', folder['file_name'])}`")
    exp_a, exp_b, exp_c = st.columns(3)
    exp_a.metric("Child folders", overview["folder_count"])
    exp_b.metric("Child files", overview["file_count"])
    exp_c.metric("Indexed descendants", len(indexer.get_descendants(selected_folder_id, include_folders=False)))

    child_items = overview["children"]
    if child_items:
        with st.expander("Immediate contents", expanded=True):
            for item in child_items:
                _result_card(item, key_prefix=f"explorer_child_{item['file_id']}", interactive=True)

    descendants = indexer.get_descendants(selected_folder_id, include_folders=False)
    if descendants:
        with st.expander(f"All files under {folder['file_name']} ({len(descendants)})", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "file_name": item["file_name"],
                            "path": item.get("path_text", ""),
                            "uploader": item.get("uploader_name", ""),
                            "type": item.get("file_type", ""),
                        }
                        for item in descendants
                    ]
                ),
                use_container_width=True,
            )


def _run_analytics_query(query: str, intent: dict, indexer: MetadataIndexer, analytics: AnalyticsEngine):
    datasets = _load_context_datasets(indexer, analytics, query=query)

    if not datasets:
        return {"text": "No tabular datasets available from the current context.", "compare_df": None, "trend_fig": None}

    action = intent.get("action", "compare")
    keywords = intent.get("keywords", []) or ["revenue", "sales", "profit"]

    if action == "trend":
        df = datasets[0]["df"]
        metric_col = analytics.find_best_column(df, keywords)
        if not metric_col:
            return {"text": "I couldn't find a numeric metric column for trend analysis.", "compare_df": None, "trend_fig": None}
        trend_fig = analytics.trend_chart(df, value_col=metric_col)
        if not trend_fig:
            return {"text": "I couldn't detect a valid date/month column for trend analysis.", "compare_df": None, "trend_fig": None}
        return {"text": f"Monthly trend generated using `{metric_col}` from `{datasets[0]['file_name']}`.", "compare_df": None, "trend_fig": trend_fig}
    if action == "dashboard":
        labels = ", ".join([item["file_name"] for item in datasets[:3]])
        return {
            "text": f"Interactive dashboard prepared below for: {labels}. Use the metric selectors, aggregation, and chart-type controls to explore KPIs, univariate, multivariate, and cross-file views.",
            "compare_df": None,
            "trend_fig": None,
        }

    compare_df = analytics.compare_metric_across_files(datasets, metric_hints=keywords)
    if compare_df.empty:
        return {"text": "I couldn't find a matching metric column across selected files.", "compare_df": None, "trend_fig": None}
    return {
        "text": f"Compared `{keywords[0] if keywords else 'metric'}` across {len(compare_df)} files.",
        "compare_df": compare_df,
        "trend_fig": None,
    }


def _run_combine_query(indexer: MetadataIndexer, analytics: AnalyticsEngine):
    datasets = _load_context_datasets(indexer, analytics, query="combine", prefer_selected=True)
    if len(datasets) < 2:
        return {"text": "Select or search at least two tabular files before combining them.", "combined_df": None}
    combined_df = analytics.combine_datasets(datasets)
    if combined_df.empty:
        return {"text": "I couldn't combine the current datasets.", "combined_df": None}
    return {"text": f"Combined {len(datasets)} datasets into one table with {len(combined_df)} rows.", "combined_df": combined_df}


def _run_folder_contents_query(query: str, indexer: MetadataIndexer):
    folders = _resolve_context_items(indexer, query, item_kind="folder")
    if not folders:
        return {"text": "I couldn't find a matching folder in the current Drive index.", "results": []}
    folder = folders[0]
    results = indexer.list_children(folder["file_id"])
    return {
        "text": f"Folder `{folder['file_name']}` contains {len(results)} indexed items.",
        "results": results,
    }


def _run_alerts_query(indexer: MetadataIndexer, analytics: AnalyticsEngine):
    datasets = _load_context_datasets(indexer, analytics, query="alerts")
    alerts = analytics.business_alerts(datasets)
    st.session_state["latest_alerts"] = alerts
    if not alerts:
        return {"text": "I didn't detect any major numeric anomalies in the current dataset context."}
    messages = [f"{item['file_name']}: {item['message']}" for item in alerts[:5]]
    return {"text": "Business alerts:\n- " + "\n- ".join(messages)}


def _run_current_dataset_query(query: str, indexer: MetadataIndexer, analytics: AnalyticsEngine):
    datasets = _load_context_datasets(indexer, analytics, query=query)
    if not datasets:
        return {
            "text": "Select a dataset first, then ask a question about the current file.",
            "table": pd.DataFrame(),
            "chart": None,
            "dataset_name": "",
        }
    target = datasets[0]
    answer = analytics.answer_dataset_question(target["df"], query)
    answer["dataset_name"] = target["file_name"]
    return answer


def _run_executive_brief(indexer: MetadataIndexer, analytics: AnalyticsEngine, ai_router: AIRouter):
    datasets = _load_context_datasets(indexer, analytics, query="executive brief")
    if not datasets:
        return {"text": "Select or search tabular files first so I can build an executive brief."}
    brief = analytics.executive_brief(datasets)
    st.session_state["executive_brief"] = brief
    if not ai_router.can_use_llm():
        return {"text": _summarize_brief_rule_based(brief)}
    prompt = (
        f"Create a concise executive brief for a {st.session_state['business_role']} audience.\n"
        f"Headlines: {json.dumps(brief['headlines'])}\n"
        f"Alerts: {json.dumps(brief['alerts'])}\n"
        f"KPIs: {json.dumps(brief['kpis'])}"
    )
    summary = ai_router.generate(prompt)
    return {"text": summary}


def _run_rename_query(query: str, indexer: MetadataIndexer, apply_changes: bool):
    items = _resolve_context_items(indexer, query)
    if not items:
        return {"text": "I couldn't find any Drive items to rename.", "preview": []}

    target_folder_name = None
    folder_match = re.search(r"(?:inside|under|in)\s+folder\s+([A-Za-z0-9 _./()-]+)", query, flags=re.IGNORECASE)
    if folder_match:
        target_folder_name = folder_match.group(1).strip(" .,!?:;")

    if target_folder_name:
        folder_matches = indexer.search_by_name(target_folder_name, item_kind="folder", limit=3)
        if folder_matches:
            items = indexer.get_descendants(folder_matches[0]["file_id"], include_folders=True)

    monitor = DriveMonitor()
    preview = monitor.preview_clean_names(items)
    preview = [item for item in preview if item["will_change"]]
    st.session_state["rename_preview"] = preview
    if not preview:
        return {"text": "Everything in the current scope already looks clean.", "preview": []}

    if not apply_changes:
        return {"text": f"I prepared {len(preview)} rename suggestions. Review them below, then ask me to apply naming cleanup.", "preview": preview}

    if st.session_state["data_source"] != "google_drive":
        return {"text": "Rename apply is only available when the app is connected to Google Drive data.", "preview": preview}

    applied = monitor.apply_rename_plan(preview)
    st.session_state["data_bootstrap_done"] = False
    if not applied:
        return {
            "text": "I prepared the rename plan, but none of the renames were applied. This usually means the service account only has Viewer access instead of Editor/Content manager on the Drive folder.",
            "preview": preview,
        }
    return {"text": f"Applied {len(applied)} Drive renames. Run sync again if you want the index refreshed immediately.", "preview": preview}


def _run_web_compare_query(query: str, indexer: MetadataIndexer, analytics: AnalyticsEngine, ai_router: AIRouter):
    domains = ["amazon.com", "amazon.in"] if "amazon" in query.lower() else None
    web_results = ai_router.search_web(query, domains=domains)
    datasets = _load_context_datasets(indexer, analytics, query=query)
    if not web_results:
        return {"text": "Web comparison needs `TAVILY_API_KEY` in secrets, or the search returned no results.", "comparison": ""}
    dataset_context = []
    for item in datasets[:3]:
        dataset_context.append(
            {
                "file_name": item["file_name"],
                "columns": list(item["df"].columns[:10]),
                "rows": len(item["df"]),
            }
        )
    prompt = (
        "Compare these Drive datasets with the web findings and highlight mismatches, opportunities, and useful benchmarks.\n"
        f"Datasets: {json.dumps(dataset_context, indent=2)}\n"
        f"Web findings: {json.dumps(web_results[:5], indent=2)}"
    )
    comparison_text = ai_router.generate(prompt) if ai_router.can_use_llm() else "Web comparison data prepared. Enable paid LLM mode for narrative synthesis."
    return {"text": "Compared current Drive context with web results.", "comparison": comparison_text}


def _run_amazon_compare_query(query: str, indexer: MetadataIndexer, analytics: AnalyticsEngine, ai_router: AIRouter):
    datasets = _load_context_datasets(indexer, analytics, query=query)
    if not datasets:
        return {"text": "Select or search a dataset first so I know what to compare against Amazon.", "comparison_df": None}

    search_term = _extract_target_name(query) or datasets[0]["file_name"].replace("_", " ")
    amazon_results = ai_router.search_amazon(search_term)
    if not amazon_results:
        return {"text": "I couldn't fetch Amazon marketplace results right now. This can happen if Amazon blocks the request or the query is too broad.", "comparison_df": None}

    comparison_df = analytics.compare_dataset_to_market(datasets[0]["df"], amazon_results)
    if comparison_df.empty:
        return {"text": "I found Amazon results, but couldn't match them to a numeric price-like column in the selected dataset.", "comparison_df": None}
    return {
        "text": (
            f"Compared `{datasets[0]['file_name']}` with {len(comparison_df)} Amazon listings. "
            f"{_summarize_market_compare_rule_based(comparison_df)}"
        ),
        "comparison_df": comparison_df,
        "raw_results": amazon_results,
    }


indexer, search_engine, analytics = get_services()
init_session()
bootstrap_data_source(indexer)
_sync_dataset_selection_from_widget()

with st.sidebar:
    st.subheader("Drive Sync")
    llm_enabled = st.toggle("Enable paid LLM", value=st.session_state["llm_enabled"])
    st.session_state["llm_enabled"] = llm_enabled
    llm_provider = st.selectbox(
        "LLM provider",
        options=["anthropic", "openai", "groq", "gemini", "huggingface"],
        index=["anthropic", "openai", "groq", "gemini", "huggingface"].index(st.session_state["llm_provider"])
        if st.session_state["llm_provider"] in ["anthropic", "openai", "groq", "gemini", "huggingface"]
        else 0,
        disabled=not llm_enabled,
    )
    st.session_state["llm_provider"] = llm_provider
    st.write(f"Provider: `{st.session_state['llm_provider']}`")
    if st.session_state["llm_call_count"] > 0:
        _tok = st.session_state["llm_token_estimate"]
        st.caption(
            f"Session LLM usage: {st.session_state['llm_call_count']} calls, "
            f"~{_tok:,} tokens (est.)"
        )
        if st.button("Reset usage counter"):
            st.session_state["llm_call_count"] = 0
            st.session_state["llm_token_estimate"] = 0
    role = st.selectbox(
        "Business role",
        options=["leadership", "sales", "finance", "marketing", "operations"],
        index=["leadership", "sales", "finance", "marketing", "operations"].index(st.session_state["business_role"])
        if st.session_state["business_role"] in ["leadership", "sales", "finance", "marketing", "operations"]
        else 0,
    )
    st.session_state["business_role"] = role
    st.caption(
        f"Auth mode: `{settings.drive_auth_mode}` | "
        f"Folder ID: `{settings.google_drive_folder_id or 'empty'}` | "
        f"Shared Drive ID: `{settings.google_shared_drive_id or 'empty'}`"
    )
    st.caption(
        f"LLM mode: `{'on' if st.session_state['llm_enabled'] else 'off'}` | "
        f"Claude key: `{'loaded' if bool(settings.anthropic_api_key) else 'missing'}` | "
        f"OpenAI key: `{'loaded' if bool(settings.openai_api_key) else 'missing'}` | "
        f"Drive credentials: `{'loaded' if bool(settings.google_service_account_json or settings.google_service_account_file) else 'missing'}` | "
        f"Web search: `{'loaded' if bool(settings.tavily_api_key) else 'missing'}`"
    )
    last_sync = st.session_state.get("last_sync_time")
    if last_sync:
        mins_ago = int((datetime.utcnow() - last_sync).total_seconds() // 60)
        st.caption(f"Last synced {mins_ago} min ago. Data stays current until you sync again.")
    if st.button("Run Monitor Once"):
        try:
            with st.spinner("Syncing Drive folder..."):
                monitor = DriveMonitor()
                changes = monitor.run_once()
            st.session_state["last_sync_time"] = datetime.utcnow()
            # Invalidate any cached datasets after a sync so stale data is not served.
            _cached_load_dataset.clear()
            st.success(
                f"Sync done. New={len(changes['new'])}, Modified={len(changes['modified'])}, Deleted={len(changes['deleted'])}"
            )
            if indexer.count_real_files() > 0:
                indexer.remove_dummy_data()
                st.session_state["data_source"] = "google_drive"
                st.session_state["data_source_detail"] = "Using Google Drive data."
        except Exception as exc:
            st.error(f"Drive sync failed: {exc}")
            _ensure_demo_data(indexer)
            st.session_state["data_source"] = "demo"
            st.session_state["data_source_detail"] = f"Drive sync failed ({exc}). Using demo data."
    if st.button("Clear history/context"):
        _reset_chat_and_context()
        st.rerun()

    st.divider()
    st.subheader("Basket")
    basket_items = [indexer.get_file(file_id) for file_id in st.session_state["selected_items"]]
    basket_items = [item for item in basket_items if item]
    if basket_items:
        for item in basket_items[:8]:
            st.write(f"- {item['file_name']}")
        if st.button("Clear basket"):
            st.session_state["selected_items"] = []
    else:
        st.caption("Add files or folders from chat results to build combined views, dashboards, and rename batches.")

    st.divider()
    st.subheader("Dashboard Presets")
    presets = _load_dashboard_presets()
    preset_names = [preset["name"] for preset in presets]
    selected_preset = st.selectbox(
        "Saved presets",
        options=[""] + preset_names,
        index=([""] + preset_names).index(st.session_state["selected_preset_name"])
        if st.session_state["selected_preset_name"] in preset_names
        else 0,
    )
    st.session_state["selected_preset_name"] = selected_preset
    load_col, delete_col = st.columns(2)
    with load_col:
        if st.button("Load preset", disabled=not selected_preset):
            preset = next((item for item in presets if item["name"] == selected_preset), None)
            if preset:
                _apply_preset_to_state(preset)
                st.success(f"Loaded preset `{selected_preset}`.")
                st.rerun()
    with delete_col:
        if st.button("Delete preset", disabled=not selected_preset):
            _delete_dashboard_preset(selected_preset)
            st.session_state["selected_preset_name"] = ""
            st.success("Preset deleted.")
            st.rerun()

ai_router = AIRouter(
    st.session_state["llm_provider"],
    enabled=st.session_state["llm_enabled"],
    prior_calls=st.session_state.get("llm_call_count", 0),
    prior_tokens=st.session_state.get("llm_token_estimate", 0),
)

if st.session_state["data_source"] == "google_drive":
    st.success("Using Google Drive data")
else:
    st.info("Using demo data")
if st.session_state["data_source_detail"]:
    st.caption(st.session_state["data_source_detail"])

all_indexed_items = indexer.list_files()
indexed_files = len([item for item in all_indexed_items if item.get("item_kind", "file") == "file"])
indexed_folders = len([item for item in all_indexed_items if item.get("item_kind") == "folder"])
top_a, top_b, top_c, top_d = st.columns(4)
top_a.metric("Indexed files", indexed_files)
top_b.metric("Indexed folders", indexed_folders)
top_c.metric("Basket items", len(st.session_state["selected_items"]))
top_d.metric("LLM mode", "On" if st.session_state["llm_enabled"] else "Off")

starter_prompts = {
    "leadership": [
        "Build an executive brief from these files",
        "Show the main business alerts in this folder",
        "Compare revenue across selected files",
    ],
    "sales": [
        "Show sales files uploaded this year",
        "Compare revenue across these files",
        "Build a dashboard for the selected sales datasets",
    ],
    "finance": [
        "Show finance folders",
        "Compare profit across these files",
        "Clean naming in folder finance",
    ],
    "marketing": [
        "Show folders available",
        "Build a dashboard for these marketing files",
        "Compare this dataset with Amazon listings",
    ],
    "operations": [
        "Show contents of folder operations",
        "What changed in the current dataset context?",
        "Clean naming in this folder",
    ],
}

with st.expander("Starter prompts", expanded=False):
    for prompt in starter_prompts.get(st.session_state["business_role"], starter_prompts["leadership"]):
        st.write(f"- {prompt}")

with st.expander("Current context", expanded=False):
    if st.session_state["selected_items"]:
        context_items = [indexer.get_file(file_id) for file_id in st.session_state["selected_items"]]
        context_items = [item for item in context_items if item]
        for item in context_items[:12]:
            st.write(f"- {item['file_name']} ({item.get('item_kind', 'file')})")
    elif st.session_state["last_results"]:
        st.caption("Using the latest search results as context.")
        for item in st.session_state["last_results"][:12]:
            st.write(f"- {item['file_name']} ({item.get('item_kind', 'file')})")
    else:
        st.caption("No active context yet. Search, select, or add items to the basket.")

st.divider()
_render_drive_explorer(indexer)

st.divider()
_render_file_actions(indexer, analytics)

st.subheader("Chat")
_show_chat_history()
user_query = st.chat_input("Ask: Show files uploaded by Amber / From these, which contain revenue data?")

if user_query:
    _sync_dataset_selection_from_widget()
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    intent = ai_router.parse_query_intent(user_query)
    filters = intent.get("filters", {}) or {}
    filters["keywords"] = intent.get("keywords", [])
    has_meaningful_filters = any(
        value not in [None, "", []]
        for value in [filters.get("uploader"), filters.get("year"), filters.get("file_type"), filters.get("item_kind"), filters.get("keywords")]
    )
    has_specific_filters = any(
        value not in [None, "", []]
        for value in [filters.get("uploader"), filters.get("year"), filters.get("file_type"), filters.get("keywords")]
    )

    lower = user_query.lower().strip()
    is_follow_up = lower.startswith("from these") or "across them" in lower

    assistant_text = ""
    st.session_state["latest_results"] = []
    st.session_state["latest_compare_df"] = None
    st.session_state["latest_alerts"] = []
    trend_fig = None
    combined_df = None
    web_comparison = ""
    market_compare_df = None
    market_raw_results = None
    rename_preview = []
    dataset_query_table = None
    dataset_query_chart = None

    if _should_answer_from_current_dataset(user_query):
        payload = _run_current_dataset_query(user_query, indexer, analytics)
        dataset_name = payload.get("dataset_name", "")
        assistant_text = f"Current dataset: `{dataset_name}`. {payload['text']}" if dataset_name else payload["text"]
        dataset_query_table = payload["table"]
        dataset_query_chart = payload["chart"]
    elif intent.get("action") in ["compare", "trend", "dashboard"]:
        payload = _run_analytics_query(user_query, intent, indexer, analytics)
        assistant_text = payload["text"]
        st.session_state["latest_compare_df"] = payload["compare_df"]
        trend_fig = payload["trend_fig"]
    elif intent.get("action") == "folder_contents":
        payload = _run_folder_contents_query(user_query, indexer)
        assistant_text = payload["text"]
        st.session_state["last_results"] = payload["results"]
        st.session_state["latest_results"] = payload["results"]
    elif intent.get("action") == "executive_brief":
        payload = _run_executive_brief(indexer, analytics, ai_router)
        assistant_text = payload["text"]
    elif intent.get("action") == "alerts":
        payload = _run_alerts_query(indexer, analytics)
        assistant_text = payload["text"]
    elif intent.get("action") == "combine":
        payload = _run_combine_query(indexer, analytics)
        assistant_text = payload["text"]
        combined_df = payload["combined_df"]
    elif intent.get("action") == "rename_preview":
        payload = _run_rename_query(user_query, indexer, apply_changes=False)
        assistant_text = payload["text"]
        rename_preview = payload["preview"]
    elif intent.get("action") == "rename_apply":
        payload = _run_rename_query(user_query, indexer, apply_changes=True)
        assistant_text = payload["text"]
        rename_preview = payload["preview"]
    elif intent.get("action") == "amazon_compare":
        payload = _run_amazon_compare_query(user_query, indexer, analytics, ai_router)
        assistant_text = payload["text"]
        market_compare_df = payload.get("comparison_df")
        market_raw_results = payload.get("raw_results")
    elif intent.get("action") == "web_compare":
        payload = _run_web_compare_query(user_query, indexer, analytics, ai_router)
        assistant_text = payload["text"]
        web_comparison = payload["comparison"]
    elif intent.get("action") == "download":
        results = _resolve_context_items(indexer, user_query, item_kind="file")
        if not results:
            results = st.session_state["last_results"] or indexer.list_items("file")[:12]
        st.session_state["latest_results"] = results
        assistant_text = f"Prepared {len(results)} file results with download actions."
    else:
        base_rows = st.session_state["last_results"] if is_follow_up else None
        should_use_builtin_search = any(
            phrase in lower
            for phrase in [
                "show folders",
                "list folders",
                "folders available",
                "what folders",
                "show files",
                "list files",
                "all files",
                "drive files",
                "what are the files",
            ]
        )
        should_use_builtin_search = should_use_builtin_search and not has_specific_filters
        results = (
            search_engine.search(user_query, top_k=12)
            if should_use_builtin_search
            else search_engine.search_with_filters(user_query, filters, top_k=12, base_rows=base_rows)
            if has_meaningful_filters
            else search_engine.search(user_query, top_k=12)
        )
        st.session_state["last_results"] = results
        st.session_state["latest_results"] = results
        st.session_state["last_selected_files"] = []
        folders = len([r for r in results if r.get("item_kind") == "folder"])
        files = len([r for r in results if r.get("item_kind", "file") == "file"])
        assistant_text = f"I found {len(results)} matching items ({files} files, {folders} folders)."
        if not results:
            assistant_text = "No matching Drive items found. Try a folder name, file name, uploader, or keyword."

    # Persist LLM usage counters back to session state so they survive reruns.
    st.session_state["llm_call_count"] = ai_router._calls
    st.session_state["llm_token_estimate"] = ai_router._tokens

    turn_id = st.session_state["chat_turn_counter"] + 1
    st.session_state["chat_turn_counter"] = turn_id
    assistant_turn = {
        "role": "assistant",
        "content": assistant_text,
        "for_query": user_query,
        "results": st.session_state["latest_results"],
        "compare_rows": st.session_state["latest_compare_df"].to_dict(orient="records")
        if st.session_state["latest_compare_df"] is not None
        else None,
        "analysis_rows": dataset_query_table.to_dict(orient="records") if isinstance(dataset_query_table, pd.DataFrame) and not dataset_query_table.empty else None,
        "rename_preview": rename_preview,
        "alerts_rows": st.session_state.get("latest_alerts"),
        "market_rows": market_compare_df.to_dict(orient="records") if market_compare_df is not None else None,
    }
    st.session_state["chat_history"].append(assistant_turn)
    with st.chat_message("assistant"):
        st.caption(f"Response to: {user_query}")
        st.write(assistant_text)

        if st.session_state["latest_results"]:
            for r in st.session_state["latest_results"]:
                _result_card(r, key_prefix=f"turn_{turn_id}", interactive=True)

            if ai_router.can_use_llm():
                compact = [
                    {
                        "file_name": r["file_name"],
                        "summary": r["summary"],
                        "keywords": json.loads(r.get("keywords", "[]")),
                        "topic": r["topic"],
                    }
                    for r in st.session_state["latest_results"][:8]
                ]
                ai_prompt = (
                    "Give concise insights from these files: top themes, trends, and anomalies to inspect.\n"
                    f"{json.dumps(compact, indent=2)}"
                )
                # Use streaming so tokens appear progressively instead of all at once.
                ai_summary = st.write_stream(ai_router.generate_stream(ai_prompt))
                # Track usage after streaming (streaming doesn't go through generate()).
                if ai_summary:
                    st.session_state["llm_call_count"] += 1
                    st.session_state["llm_token_estimate"] += (len(ai_prompt) + len(ai_summary)) // 4
            else:
                st.write(_summarize_results_rule_based(st.session_state["latest_results"]))

        if st.session_state["latest_compare_df"] is not None:
            compare_df = st.session_state["latest_compare_df"]
            st.dataframe(compare_df, use_container_width=True)
            fig = analytics.compare_chart(compare_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if dataset_query_table is not None and not dataset_query_table.empty:
            st.dataframe(dataset_query_table, use_container_width=True)
        if dataset_query_chart is not None:
            st.plotly_chart(dataset_query_chart, use_container_width=True)

        if trend_fig is not None:
            st.plotly_chart(trend_fig, use_container_width=True)

        if combined_df is not None:
            st.dataframe(combined_df, use_container_width=True)
            export_name = f"combined_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_bytes = combined_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download combined dataset",
                data=csv_bytes,
                file_name=export_name,
                mime="text/csv",
                key="download_combined_dataset",
            )
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                combined_df.to_excel(writer, index=False, sheet_name="CombinedData")
            st.download_button(
                "Download combined dataset (Excel)",
                data=excel_buffer.getvalue(),
                file_name=export_name.replace(".csv", ".xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_combined_dataset_excel",
            )

        if rename_preview:
            st.dataframe(pd.DataFrame(rename_preview), use_container_width=True)

        if st.session_state.get("latest_alerts"):
            st.dataframe(pd.DataFrame(st.session_state["latest_alerts"]), use_container_width=True)

        if st.session_state.get("executive_brief"):
            brief = st.session_state["executive_brief"]
            if brief.get("kpis"):
                st.dataframe(pd.DataFrame(brief["kpis"]), use_container_width=True)
            if brief.get("alerts"):
                st.dataframe(pd.DataFrame(brief["alerts"]), use_container_width=True)

        if market_compare_df is not None:
            st.dataframe(market_compare_df, use_container_width=True)
            if not market_compare_df.empty:
                fig = analytics.comparison_chart(market_compare_df, category_col="market_item", value_col="market_price")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        if market_raw_results:
            st.write("Amazon matches")
            st.dataframe(pd.DataFrame(market_raw_results), use_container_width=True)

        if web_comparison:
            st.write(web_comparison)

    _scroll_chat_to_bottom()


st.divider()
st.subheader("Business Overview")
current_context_datasets = _load_context_datasets(indexer, analytics)
if current_context_datasets:
    brief = analytics.executive_brief(current_context_datasets)
    total_rows = sum(kpi.get("rows", 0) for kpi in brief["kpis"])
    total_alerts = len(brief["alerts"])
    context_files = len(current_context_datasets)
    col1, col2, col3 = st.columns(3)
    col1.metric("Datasets in context", context_files)
    col2.metric("Total rows", f"{total_rows:,}")
    col3.metric("Active alerts", total_alerts)
    st.markdown("#### What This Context Contains")
    for headline in brief["headlines"][:5]:
        st.write(f"- {headline}")
    if brief["alerts"]:
        st.markdown("#### Priority Alerts")
        for alert in brief["alerts"][:5]:
            st.warning(f"{alert['file_name']}: {alert['message']}")
    else:
        st.caption("No high-priority alert signals were detected in the current context.")
    with st.expander("KPI table", expanded=False):
        kpi_df = pd.DataFrame(brief["kpis"])
        if not kpi_df.empty:
            st.dataframe(kpi_df, use_container_width=True)
    st.write(_summarize_brief_rule_based(brief))
else:
    st.caption("Add files to the basket or select datasets below to generate executive KPIs and alerts.")

st.divider()
st.subheader("Interactive Dashboard")
dashboard_context = _load_context_datasets(indexer, analytics, query="selected dashboard", prefer_selected=True)
if dashboard_context:
    _render_cross_file_dashboard(dashboard_context, analytics, key_prefix="global_dashboard")
    for dashboard_item in dashboard_context[:3]:
        st.markdown(f"### Dashboard: {dashboard_item['file_name']}")
        _render_dataset_dashboard(
            dashboard_item["df"],
            analytics,
            file_label=dashboard_item["file_name"],
            key_prefix=f"dashboard_{dashboard_item['info']['file_id']}",
            dataset_id=dashboard_item["info"]["file_id"],
        )
else:
    st.caption("Select one or more datasets below, or add files to the basket, to open the interactive dashboard builder.")

st.divider()
st.subheader("Dataset Details")
_ensure_demo_data(indexer)
all_files = indexer.list_files()
tabular_files = [f for f in all_files if Path(f.get("local_path", "")).suffix.lower() in [".csv", ".xlsx", ".xls"]]
selected = st.multiselect(
    "Select one or more datasets",
    options=[f"{f['file_id']} | {f['file_name']}" for f in tabular_files],
    key="dataset_selection",
)

if selected:
    _sync_dataset_selection_from_widget()
    for item in selected:
        file_id = item.split("|")[0].strip()
        info = indexer.get_file(file_id)
        if not info:
            continue
        st.markdown(f"### {info['file_name']}")
        try:
            df = analytics.load_dataset(info["local_path"])
        except Exception as exc:
            st.error(f"Failed to load `{info['file_name']}`: {exc}")
            continue

        profile = analytics.dataset_profile(df)
        default_bundle = analytics.default_dashboard_bundle(df)
        default_panels = default_bundle.get("panels", [])
        dataset_type = default_bundle.get("dataset_type", analytics.infer_dataset_type(df))
        metric_definition_map = {
            item["metric"]: item["definition"] for item in default_bundle.get("metric_definitions", [])
        }
        dimensions = default_bundle["dimensions"]
        primary_metric = default_bundle["metrics"][0] if default_bundle["metrics"] else analytics.find_best_column(df, ["revenue", "sales", "profit"])
        grouped_anomalies = default_bundle.get("grouped_anomalies", {"count": 0, "rows": [], "method": "", "bounds": {}})
        anomaly_count = grouped_anomalies["count"] if grouped_anomalies.get("count") else 0
        dataset_anchor_base = _slugify(file_id)
        _render_dataset_toolbar(
            file_name=info["file_name"],
            dataset_type=dataset_type,
            row_count=profile["row_count"],
            column_count=profile["column_count"],
            primary_metric=analytics.humanize_label(primary_metric) if primary_metric else None,
            time_dim=analytics.humanize_label(dimensions["time"]) if dimensions.get("time") else None,
            category_dim=analytics.humanize_label(dimensions["category"]) if dimensions.get("category") else None,
        )
        _render_dataset_mini_nav(dataset_anchor_base)
        _render_anchor(f"{dataset_anchor_base}-summary")
        st.markdown(analytics.explain_table(df))
        st.caption(default_bundle.get("narrative", ""))

        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        overview_col1.metric("Rows", f"{profile['row_count']:,}")
        overview_col2.metric("Columns", profile["column_count"])
        overview_col3.metric("Dataset type", dataset_type.title())
        overview_col4.metric("Grouped anomalies", anomaly_count)

        with st.expander("Column guide", expanded=False):
            st.caption(f"Best time dimension: {dimensions['time'] or 'Not detected'}")
            st.caption(f"Best category dimension: {dimensions['category'] or 'Not detected'}")
            _render_column_guide(profile)

        _render_anchor(f"{dataset_anchor_base}-metric-guide")
        with st.expander("Metric guide", expanded=False):
            st.caption("These definitions help users understand what each suggested metric represents in business terms.")
            _render_metric_definitions(default_bundle.get("metric_definitions", []))

        benchmark_cards = analytics.benchmark_cards(
            df,
            primary_metric,
            category_col=dimensions.get("category"),
            date_col=dimensions.get("time"),
        )
        if benchmark_cards:
            _render_anchor(f"{dataset_anchor_base}-benchmarks")
            st.markdown("#### Benchmark Cards")
            _render_benchmark_cards(benchmark_cards)

        _render_anchor(f"{dataset_anchor_base}-full-table")
        with st.expander("Full table", expanded=True):
            st.caption("Showing the full cleaned dataset below. Scroll horizontally and vertically as needed.")
            st.dataframe(df, use_container_width=True, height=620)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download cleaned CSV for {info['file_name']}",
                data=csv_bytes,
                file_name=f"{Path(info['file_name']).stem}_cleaned.csv",
                mime="text/csv",
                key=f"clean_csv_{file_id}",
            )

        if default_bundle["kpis"]:
            _render_anchor(f"{dataset_anchor_base}-kpis")
            st.markdown("#### KPI Snapshot")
            _render_kpi_cards(default_bundle["kpis"], key_prefix=f"kpi_{file_id}", definitions=metric_definition_map, analytics=analytics)

        if default_panels:
            _render_anchor(f"{dataset_anchor_base}-best-dashboard")
            st.markdown("#### Best Dashboard")
            st.caption(
                f"Recommended view for this {dataset_type} dataset uses metrics {', '.join(default_bundle['metrics'][:3]) or 'none'}, "
                f"time dimension `{dimensions['time'] or 'not detected'}`, and category dimension `{dimensions['category'] or 'not detected'}`."
            )
            with st.expander("Why this dashboard was chosen", expanded=False):
                _render_chart_reasons(default_bundle.get("chart_reasons", []))
            _render_chart_panels(default_panels, dataset_id=file_id, section_key="best_dashboard")

        _render_anchor(f"{dataset_anchor_base}-builder")
        _render_dataset_dashboard(
            df,
            analytics,
            file_label=info["file_name"],
            key_prefix=f"details_{file_id}",
            dataset_id=file_id,
        )

        if primary_metric:
            _render_anchor(f"{dataset_anchor_base}-anomalies")
            st.markdown("#### Anomaly Review")
            st.caption(
                "Use grouped anomaly detection to review unusual ASINs, dates, or categories instead of only looking at individual rows."
            )
            _render_grouped_anomaly_review(df, analytics, default_bundle, file_id, primary_metric)

        metric_summary = analytics.compute_metrics(df, target_col=primary_metric) if primary_metric else {"rows": len(df), "columns": len(df.columns)}
        if ai_router.can_use_llm():
            ai_insight_prompt = (
                f"Dataset: {info['file_name']}\n"
                f"Metric column: {primary_metric}\n"
                f"Stats: {json.dumps(metric_summary)}\n"
                f"Anomaly explanation: {grouped_anomalies.get('method', '')}\n"
                "Provide 3 concise insights: what the table is about, what metric matters, and what risk/opportunity to check."
            )
            insight_text = ai_router.generate(ai_insight_prompt)
            if insight_text:
                st.markdown("#### Plain-language summary")
                st.write(insight_text)
        else:
            st.markdown("#### Plain-language summary")
            st.write(_summarize_dataset_rule_based(info["file_name"], metric_summary, grouped_anomalies, analytics.insights(df)))

        export_name = f"analysis_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        export_path = str(EXPORT_DIR / export_name)
        _render_anchor(f"{dataset_anchor_base}-exports")
        st.markdown("#### Export Options")
        if default_panels:
            pdf_bytes = _dashboard_pdf_bytes(info["file_name"], default_panels)
            if pdf_bytes:
                st.download_button(
                    f"Download dashboard PDF for {info['file_name']}",
                    data=pdf_bytes,
                    file_name=f"{Path(info['file_name']).stem}_dashboard.pdf",
                    mime="application/pdf",
                    key=f"dashboard_pdf_{file_id}",
                )
            else:
                st.caption("Dashboard PDF export is unavailable until Plotly image export support is available in deployment.")
        try:
            insights = analytics.insights(df)
            analytics.export_report(df, metric_summary, insights, export_path)
            with open(export_path, "rb") as f:
                st.download_button(
                    f"Download report for {info['file_name']}",
                    data=f.read(),
                    file_name=export_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{file_id}",
                )
        except Exception as exc:
            st.error(f"Export failed for `{info['file_name']}`: {exc}")
