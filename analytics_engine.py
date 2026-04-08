from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class AnalyticsEngine:
    def load_dataset(self, local_path: str) -> pd.DataFrame:
        path = Path(local_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported dataset type for analytics: {suffix}")

    def compute_metrics(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        metrics = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "numeric_columns": numeric_cols,
        }
        if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            metrics["sum"] = float(df[target_col].sum())
            metrics["average"] = float(df[target_col].mean())
        elif numeric_cols:
            col = numeric_cols[0]
            metrics["default_metric_column"] = col
            metrics["sum"] = float(df[col].sum())
            metrics["average"] = float(df[col].mean())
        return metrics

    def dataset_summary(self, df: pd.DataFrame) -> Dict:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        return {
            "rows": int(len(df)),
            "columns": [str(c) for c in df.columns],
            "numeric_columns": numeric_cols,
            "sample": df.head(5).to_dict(orient="records"),
        }

    def find_best_column(self, df: pd.DataFrame, hints: Optional[List[str]] = None) -> Optional[str]:
        hints = [h.lower() for h in (hints or [])]
        cols = [str(c) for c in df.columns]
        for hint in hints:
            for col in cols:
                if hint in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
                    return col
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        return numeric_cols[0] if numeric_cols else None

    def detect_anomalies(self, df: pd.DataFrame, value_col: str, date_col: Optional[str] = None) -> Dict:
        if value_col not in df.columns or not pd.api.types.is_numeric_dtype(df[value_col]):
            return {"count": 0, "rows": []}
        q1 = df[value_col].quantile(0.25)
        q3 = df[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[value_col] < lower) | (df[value_col] > upper)].copy()
        if date_col and date_col in outliers.columns:
            outliers[date_col] = pd.to_datetime(outliers[date_col], errors="coerce")
        return {"count": int(len(outliers)), "rows": outliers.head(20).to_dict(orient="records")}

    def compare_metric_across_files(self, datasets: List[Dict], metric_hints: Optional[List[str]] = None) -> pd.DataFrame:
        rows = []
        hints = metric_hints or ["revenue", "sales", "profit"]
        for item in datasets:
            file_name = item["file_name"]
            df = item["df"]
            metric_col = self.find_best_column(df, hints)
            if not metric_col:
                continue
            rows.append(
                {
                    "file_name": file_name,
                    "metric_column": metric_col,
                    "sum": float(df[metric_col].sum()),
                    "average": float(df[metric_col].mean()),
                }
            )
        return pd.DataFrame(rows)

    def compare_chart(self, compare_df: pd.DataFrame):
        if compare_df.empty:
            return None
        fig = go.Figure()
        fig.add_bar(x=compare_df["file_name"], y=compare_df["sum"], name="Sum")
        fig.add_bar(x=compare_df["file_name"], y=compare_df["average"], name="Average")
        fig.update_layout(barmode="group", title="Cross-file Comparison")
        return fig

    def combine_datasets(self, datasets: List[Dict]) -> pd.DataFrame:
        frames = []
        for item in datasets:
            df = item["df"].copy()
            df["source_file"] = item["file_name"]
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    def build_kpi_snapshot(self, df: pd.DataFrame, metric_hints: Optional[List[str]] = None) -> Dict:
        metric_col = self.find_best_column(df, metric_hints or ["revenue", "sales", "profit", "price"])
        snapshot = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "metric_column": metric_col,
        }
        if not metric_col:
            return snapshot
        snapshot["total"] = float(df[metric_col].sum())
        snapshot["average"] = float(df[metric_col].mean())
        snapshot["max"] = float(df[metric_col].max())
        snapshot["min"] = float(df[metric_col].min())
        return snapshot

    def business_alerts(self, datasets: List[Dict], metric_hints: Optional[List[str]] = None) -> List[Dict]:
        alerts = []
        for item in datasets:
            df = item["df"]
            metric_col = self.find_best_column(df, metric_hints or ["revenue", "sales", "profit", "price"])
            if not metric_col:
                continue
            anomalies = self.detect_anomalies(df, metric_col)
            if anomalies["count"] > 0:
                alerts.append(
                    {
                        "file_name": item["file_name"],
                        "severity": "high" if anomalies["count"] >= 3 else "medium",
                        "message": f"{anomalies['count']} anomaly rows detected in `{metric_col}`.",
                    }
                )
            if len(df) > 1:
                latest = df[metric_col].iloc[-1]
                baseline = df[metric_col].iloc[0]
                if baseline not in [0, None]:
                    change_pct = ((latest - baseline) / baseline) * 100
                    if abs(change_pct) >= 15:
                        direction = "up" if change_pct > 0 else "down"
                        alerts.append(
                            {
                                "file_name": item["file_name"],
                                "severity": "medium",
                                "message": f"{metric_col} moved {direction} {abs(change_pct):.1f}% across the visible period.",
                            }
                        )
        return alerts

    def executive_brief(self, datasets: List[Dict], metric_hints: Optional[List[str]] = None) -> Dict:
        headlines = []
        kpis = []
        for item in datasets:
            df = item["df"]
            metric_col = self.find_best_column(df, metric_hints or ["revenue", "sales", "profit", "price"])
            if not metric_col:
                continue
            snapshot = self.build_kpi_snapshot(df, metric_hints)
            snapshot["file_name"] = item["file_name"]
            kpis.append(snapshot)
            headlines.append(
                f"{item['file_name']}: total {metric_col} {snapshot.get('total', 0):,.2f}, "
                f"average {snapshot.get('average', 0):,.2f}."
            )
        return {
            "headlines": headlines[:5],
            "alerts": self.business_alerts(datasets, metric_hints)[:5],
            "kpis": kpis,
        }

    def compare_dataset_to_market(self, df: pd.DataFrame, external_rows: List[Dict], metric_hints: Optional[List[str]] = None) -> pd.DataFrame:
        metric_col = self.find_best_column(df, metric_hints or ["price", "revenue", "sales", "profit"])
        if not metric_col or not external_rows:
            return pd.DataFrame()

        internal_average = float(df[metric_col].mean())
        comparison_rows = []
        for row in external_rows:
            raw_price = str(row.get("price", "")).replace(",", "").strip()
            try:
                market_price = float(raw_price)
            except ValueError:
                continue
            comparison_rows.append(
                {
                    "market_item": row.get("title", "Unknown"),
                    "market_price": market_price,
                    "internal_average": internal_average,
                    "price_gap": market_price - internal_average,
                    "source": row.get("source", "market"),
                    "url": row.get("url", ""),
                }
            )
        return pd.DataFrame(comparison_rows)

    def trend_chart(
        self, df: pd.DataFrame, date_col: Optional[str] = None, value_col: Optional[str] = None
    ):
        date_candidate = date_col
        if not date_candidate:
            for c in df.columns:
                if "date" in c.lower() or "month" in c.lower():
                    date_candidate = c
                    break
        value_candidate = value_col
        if not value_candidate:
            nums = df.select_dtypes(include="number").columns.tolist()
            value_candidate = nums[0] if nums else None
        if not date_candidate or not value_candidate:
            return None
        local = df.copy()
        local[date_candidate] = pd.to_datetime(local[date_candidate], errors="coerce")
        local = local.dropna(subset=[date_candidate, value_candidate]).sort_values(date_candidate)
        if local.empty:
            return None
        return px.line(local, x=date_candidate, y=value_candidate, title="Trend Analysis")

    def comparison_chart(self, df: pd.DataFrame, category_col: Optional[str] = None, value_col: Optional[str] = None):
        cat = category_col
        if not cat:
            for c in df.columns:
                if df[c].dtype == "object":
                    cat = c
                    break
        val = value_col
        if not val:
            nums = df.select_dtypes(include="number").columns.tolist()
            val = nums[0] if nums else None
        if not cat or not val:
            return None
        grouped = df.groupby(cat, as_index=False)[val].sum().sort_values(val, ascending=False).head(15)
        return px.bar(grouped, x=cat, y=val, title="Category Comparison")

    def insights(self, df: pd.DataFrame) -> List[str]:
        insights = []
        nums = df.select_dtypes(include="number").columns.tolist()
        if nums:
            col = nums[0]
            max_idx = df[col].idxmax() if not df.empty else None
            if max_idx is not None:
                insights.append(f"Highest `{col}` = {df.loc[max_idx, col]}")
            insights.append(f"Average `{col}` = {df[col].mean():,.2f}")
        if any("month" in c.lower() for c in df.columns):
            mcol = [c for c in df.columns if "month" in c.lower()][0]
            insights.append(f"Detected monthly structure in column `{mcol}`.")
        return insights

    def export_report(self, df: pd.DataFrame, metrics: Dict, insights: List[str], output_path: str) -> str:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="RawData")
            pd.DataFrame([metrics]).to_excel(writer, index=False, sheet_name="Metrics")
            pd.DataFrame({"insights": insights}).to_excel(writer, index=False, sheet_name="Insights")
        return output_path
