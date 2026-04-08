from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_router import AIRouter
from analytics_engine import AnalyticsEngine
from config import EXPORT_DIR, settings
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
    ai = AIRouter(settings.ai_provider)
    return indexer, search, analytics, ai


def init_session() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_results", [])
    st.session_state.setdefault("last_selected_files", [])
    st.session_state.setdefault("latest_results", [])
    st.session_state.setdefault("latest_compare_df", None)
    st.session_state.setdefault("data_source", "unknown")
    st.session_state.setdefault("data_source_detail", "")
    st.session_state.setdefault("data_bootstrap_done", False)


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


def bootstrap_data_source(indexer: MetadataIndexer) -> None:
    if st.session_state.get("data_bootstrap_done"):
        return

    source = "demo"
    detail = ""
    try:
        monitor = DriveMonitor()
        monitor.run_once()
        real_count = indexer.count_real_files()
        if real_count > 0:
            indexer.remove_dummy_data()
            source = "google_drive"
            detail = f"Using Google Drive data ({real_count} files indexed)."
        else:
            seeded = _ensure_demo_data(indexer)
            source = "demo"
            detail = (
                "Drive folder is empty. Using demo data."
                if seeded > 0
                else "Drive folder is empty. No records available."
            )
    except Exception as exc:
        seeded = _ensure_demo_data(indexer)
        source = "demo"
        detail = (
            f"Drive unavailable ({exc}). Using demo data."
            if seeded > 0
            else f"Drive unavailable ({exc}). No data available."
        )

    st.session_state["data_source"] = source
    st.session_state["data_source_detail"] = detail
    st.session_state["data_bootstrap_done"] = True


def _result_card(r: dict) -> None:
    keywords = ", ".join(json.loads(r.get("keywords", "[]")))
    link = r.get("file_link", "")
    if link:
        link_line = f"- [Open file]({link})"
    else:
        link_line = "- Local/demo file (no Drive link)"
    st.markdown(
        f"**{r['file_name']}**  \n"
        f"- Uploader: `{r['uploader_name']}`  \n"
        f"- Type: `{r['file_type']}`  \n"
        f"- Modified: `{r['modified_time']}`  \n"
        f"- Topic: `{r['topic']}`  \n"
        f"- Keywords: `{keywords}`  \n"
        f"- Similarity score: `{r.get('score', '-')}`  \n"
        f"{link_line}"
    )
    st.write(r["summary"])


def _show_chat_history() -> None:
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


def _run_analytics_query(query: str, intent: dict, indexer: MetadataIndexer, analytics: AnalyticsEngine):
    file_ids = st.session_state["last_selected_files"] or [r["file_id"] for r in st.session_state["last_results"]]
    datasets = []
    for fid in file_ids:
        info = indexer.get_file(fid)
        if not info:
            continue
        suffix = Path(info.get("local_path", "")).suffix.lower()
        if suffix not in [".csv", ".xlsx", ".xls"]:
            continue
        try:
            datasets.append({"file_name": info["file_name"], "df": analytics.load_dataset(info["local_path"])})
        except Exception:
            continue

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

    compare_df = analytics.compare_metric_across_files(datasets, metric_hints=keywords)
    if compare_df.empty:
        return {"text": "I couldn't find a matching metric column across selected files.", "compare_df": None, "trend_fig": None}
    return {
        "text": f"Compared `{keywords[0] if keywords else 'metric'}` across {len(compare_df)} files.",
        "compare_df": compare_df,
        "trend_fig": None,
    }


indexer, search_engine, analytics, ai_router = get_services()
init_session()
bootstrap_data_source(indexer)

with st.sidebar:
    st.subheader("Drive Sync")
    st.write(f"Provider: `{settings.ai_provider}`")
    st.caption(
        f"Auth mode: `{settings.drive_auth_mode}` | "
        f"Folder ID: `{settings.google_drive_folder_id or 'empty'}` | "
        f"Shared Drive ID: `{settings.google_shared_drive_id or 'empty'}`"
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

if st.session_state["data_source"] == "google_drive":
    st.success("Using Google Drive data")
else:
    st.info("Using demo data")
if st.session_state["data_source_detail"]:
    st.caption(st.session_state["data_source_detail"])

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

    lower = user_query.lower().strip()
    is_follow_up = lower.startswith("from these") or "across them" in lower

    assistant_text = ""
    st.session_state["latest_results"] = []
    st.session_state["latest_compare_df"] = None
    trend_fig = None

    if intent.get("action") in ["compare", "trend"]:
        payload = _run_analytics_query(user_query, intent, indexer, analytics)
        assistant_text = payload["text"]
        st.session_state["latest_compare_df"] = payload["compare_df"]
        trend_fig = payload["trend_fig"]
    else:
        base_rows = st.session_state["last_results"] if is_follow_up else None
        results = search_engine.search_with_filters(user_query, filters, top_k=12, base_rows=base_rows)
        st.session_state["last_results"] = results
        st.session_state["latest_results"] = results
        st.session_state["last_selected_files"] = [r["file_id"] for r in results[:5]]
        assistant_text = f"I found {len(results)} matching files."
        if not results:
            assistant_text = "No matching files found. Try changing uploader, year, or keyword."

    st.session_state["chat_history"].append({"role": "assistant", "content": assistant_text})
    with st.chat_message("assistant"):
        st.write(assistant_text)

        if st.session_state["latest_results"]:
            for r in st.session_state["latest_results"]:
                _result_card(r)

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
            st.write(ai_summary)

        if st.session_state["latest_compare_df"] is not None:
            compare_df = st.session_state["latest_compare_df"]
            st.dataframe(compare_df, use_container_width=True)
            fig = analytics.compare_chart(compare_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if trend_fig is not None:
            st.plotly_chart(trend_fig, use_container_width=True)


st.divider()
st.subheader("Dataset Details")
all_files = indexer.list_files()
tabular_files = [f for f in all_files if Path(f.get("local_path", "")).suffix.lower() in [".csv", ".xlsx", ".xls"]]
selected = st.multiselect(
    "Select one or more datasets",
    options=[f"{f['file_id']} | {f['file_name']}" for f in tabular_files],
)

if selected:
    st.session_state["last_selected_files"] = [item.split("|")[0].strip() for item in selected]
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
        st.write(
            {
                "rows": summary["rows"],
                "column_count": len(summary["columns"]),
                "columns": summary["columns"][:12],
            }
        )
        st.dataframe(df.head(50), use_container_width=True)

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

        ai_insight_prompt = (
            f"Dataset: {info['file_name']}\n"
            f"Metric column: {metric_col}\n"
            f"Stats: {json.dumps(metrics)}\n"
            f"Anomaly count: {anomalies['count']}\n"
            "Provide 3 concise insights: trend, top category/pattern, and risk/anomaly."
        )
        st.write(ai_router.generate(ai_insight_prompt))

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
