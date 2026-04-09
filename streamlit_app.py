from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from ai_router import AIRouter
from analytics_engine import AnalyticsEngine
from config import DASHBOARD_PRESETS_PATH, EXPORT_DIR, settings
from metadata_indexer import MetadataIndexer
from monitor_drive import DriveMonitor
from semantic_search import SemanticSearchEngine


st.set_page_config(page_title="Drive AI Assistant", layout="wide")
st.title("Drive AI Assistant")
st.caption("Conversational Drive intelligence with memory, analytics, and exports")


@st.cache_resource
def get_services():
    indexer = MetadataIndexer()
    search = SemanticSearchEngine(indexer)
    analytics = AnalyticsEngine()
    return indexer, search, analytics


def init_session() -> None:
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


def _load_dashboard_presets() -> List[Dict]:
    if not DASHBOARD_PRESETS_PATH.exists():
        return []
    try:
        data = json.loads(DASHBOARD_PRESETS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


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
        try:
            datasets.append({"file_name": info["file_name"], "df": analytics.load_dataset(info["local_path"]), "info": info})
        except Exception:
            continue
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
        lines.append(f"{anomalies['count']} anomaly rows need review.")
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


def _render_kpi_cards(kpis: List[Dict], key_prefix: str) -> None:
    if not kpis:
        st.caption("No numeric KPI metrics available.")
        return
    cols = st.columns(min(4, len(kpis)))
    for idx, kpi in enumerate(kpis[:4]):
        with cols[idx]:
            st.metric(kpi["metric"], f"{kpi['sum']:,.2f}", help=f"Avg: {kpi['average']:,.2f} | Median: {kpi['median']:,.2f}")


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
    st.write(
        {
            "rows": profile["row_count"],
            "column_count": profile["column_count"],
            "numeric_columns": profile["numeric_columns"][:8],
            "categorical_columns": profile["categorical_columns"][:8],
            "datetime_columns": profile["datetime_columns"][:8],
        }
    )

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
        _render_kpi_cards(metric_kpis, key_prefix=f"{key_prefix}_kpi")

        grouped_preview = analytics.aggregate_metrics(df, selected_metrics, group_col=date_col or category_col, agg=agg)
        if not grouped_preview.empty:
            st.caption("Aggregated preview")
            st.dataframe(grouped_preview.head(20), use_container_width=True)

        figures = analytics.dashboard_figures(
            df=df,
            metrics=selected_metrics,
            category_col=category_col,
            date_col=date_col,
            chart_types=chart_types,
            agg=agg,
        )
        if not figures:
            st.info("No compatible charts could be generated with the current selections. Try a different metric or dimension.")
        for fig in figures:
            st.plotly_chart(fig, use_container_width=True)


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
    st.write(
        {
            "folder": folder["file_name"],
            "path": folder.get("path_text", folder["file_name"]),
            "child_folders": overview["folder_count"],
            "child_files": overview["file_count"],
        }
    )

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
    if st.button("Run Monitor Once"):
        try:
            with st.spinner("Syncing Drive folder..."):
                monitor = DriveMonitor()
                changes = monitor.run_once()
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

ai_router = AIRouter(st.session_state["llm_provider"], enabled=st.session_state["llm_enabled"])

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

st.subheader("Chat")
_show_chat_history()
user_query = st.chat_input("Ask: Show files uploaded by Amber / From these, which contain revenue data?")

if user_query:
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

    if intent.get("action") in ["compare", "trend", "dashboard"]:
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
                ai_summary = ai_router.generate(ai_prompt)
                if ai_summary:
                    st.write(ai_summary)
            else:
                st.write(_summarize_results_rule_based(st.session_state["latest_results"]))

        if st.session_state["latest_compare_df"] is not None:
            compare_df = st.session_state["latest_compare_df"]
            st.dataframe(compare_df, use_container_width=True)
            fig = analytics.compare_chart(compare_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

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
    kpi_df = pd.DataFrame(brief["kpis"])
    if not kpi_df.empty:
        st.dataframe(kpi_df, use_container_width=True)
    alerts_df = pd.DataFrame(brief["alerts"])
    if not alerts_df.empty:
        st.dataframe(alerts_df, use_container_width=True)
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
)

if selected:
    st.session_state["last_selected_files"] = [item.split("|")[0].strip() for item in selected]
    st.session_state["selected_items"] = list(dict.fromkeys(st.session_state["selected_items"] + st.session_state["last_selected_files"]))
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

        summary = analytics.dataset_summary(df)
        profile = analytics.dataset_profile(df)
        st.write(
            {
                "rows": summary["rows"],
                "column_count": len(summary["columns"]),
                "numeric_columns": profile["numeric_columns"][:8],
                "categorical_columns": profile["categorical_columns"][:8],
                "suggested_metrics": profile["metric_suggestions"][:6],
            }
        )
        st.dataframe(df.head(50), use_container_width=True)
        _render_dataset_dashboard(
            df,
            analytics,
            file_label=info["file_name"],
            key_prefix=f"details_{file_id}",
            dataset_id=file_id,
        )
        metric_col = analytics.find_best_column(df, ["revenue", "sales", "profit"])
        if not metric_col:
            st.warning("No numeric metric column found for this dataset.")
            continue

        metrics = analytics.compute_metrics(df, target_col=metric_col)
        st.json(metrics)
        trend = analytics.trend_chart(df, value_col=metric_col)
        if trend:
            st.plotly_chart(trend, use_container_width=True)
        bar = analytics.comparison_chart(df, value_col=metric_col)
        if bar:
            st.plotly_chart(bar, use_container_width=True)

        anomalies = analytics.detect_anomalies(df, metric_col)
        st.write(f"Anomalies detected: {anomalies['count']}")
        if anomalies["rows"]:
            st.dataframe(pd.DataFrame(anomalies["rows"]), use_container_width=True)

        if ai_router.can_use_llm():
            ai_insight_prompt = (
                f"Dataset: {info['file_name']}\n"
                f"Metric column: {metric_col}\n"
                f"Stats: {json.dumps(metrics)}\n"
                f"Anomaly count: {anomalies['count']}\n"
                "Provide 3 concise insights: trend, top category/pattern, and risk/anomaly."
            )
            insight_text = ai_router.generate(ai_insight_prompt)
            if insight_text:
                st.write(insight_text)
        else:
            st.write(_summarize_dataset_rule_based(info["file_name"], metrics, anomalies, analytics.insights(df)))

        export_name = f"analysis_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        export_path = str(EXPORT_DIR / export_name)
        try:
            insights = analytics.insights(df)
            analytics.export_report(df, metrics, insights, export_path)
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
