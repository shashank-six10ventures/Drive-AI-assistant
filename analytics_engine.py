from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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

    def dataset_profile(self, df: pd.DataFrame) -> Dict:
        numeric_cols = [str(col) for col in df.select_dtypes(include="number").columns.tolist()]
        categorical_cols = [str(col) for col in df.columns if str(col) not in numeric_cols]
        datetime_cols = []
        for col in df.columns:
            col_name = str(col)
            if "date" in col_name.lower() or "month" in col_name.lower() or "year" in col_name.lower():
                datetime_cols.append(col_name)
                continue
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().mean() > 0.7:
                    datetime_cols.append(col_name)
            except Exception:
                continue
        datetime_cols = list(dict.fromkeys(datetime_cols))
        categorical_cols = [col for col in categorical_cols if col not in datetime_cols]
        metric_suggestions = [
            col
            for col in numeric_cols
            if any(token in col.lower() for token in ["revenue", "sales", "profit", "price", "cost", "qty", "amount"])
        ] or numeric_cols[:4]
        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "metric_suggestions": metric_suggestions,
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

    def common_numeric_columns(self, datasets: List[Dict]) -> List[str]:
        if not datasets:
            return []
        numeric_sets = []
        for item in datasets:
            numeric_sets.append(set(item["df"].select_dtypes(include="number").columns.tolist()))
        common = set.intersection(*numeric_sets) if numeric_sets else set()
        return sorted([str(col) for col in common])

    def compare_metrics_across_files(self, datasets: List[Dict], metrics: List[str], agg: str = "sum") -> pd.DataFrame:
        rows = []
        for item in datasets:
            df = item["df"]
            record = {"file_name": item["file_name"]}
            for metric in metrics:
                if metric not in df.columns or not pd.api.types.is_numeric_dtype(df[metric]):
                    continue
                series = df[metric].dropna()
                if series.empty:
                    continue
                if agg == "mean":
                    record[metric] = float(series.mean())
                elif agg == "median":
                    record[metric] = float(series.median())
                elif agg == "min":
                    record[metric] = float(series.min())
                elif agg == "max":
                    record[metric] = float(series.max())
                elif agg == "count":
                    record[metric] = float(series.count())
                else:
                    record[metric] = float(series.sum())
            rows.append(record)
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

    def metric_kpis(self, df: pd.DataFrame, metric_cols: List[str]) -> List[Dict]:
        kpis = []
        for metric in metric_cols:
            if metric not in df.columns or not pd.api.types.is_numeric_dtype(df[metric]):
                continue
            series = df[metric].dropna()
            if series.empty:
                continue
            kpis.append(
                {
                    "metric": metric,
                    "sum": float(series.sum()),
                    "average": float(series.mean()),
                    "median": float(series.median()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )
        return kpis

    def aggregate_metrics(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        group_col: Optional[str] = None,
        agg: str = "sum",
    ) -> pd.DataFrame:
        valid_metrics = [metric for metric in metrics if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric])]
        if not valid_metrics:
            return pd.DataFrame()
        if not group_col or group_col not in df.columns:
            return pd.DataFrame([{metric: getattr(df[metric], agg)() if hasattr(df[metric], agg) else df[metric].sum() for metric in valid_metrics}])

        local = df.copy()
        if group_col:
            if group_col.lower().endswith("date") or "month" in group_col.lower() or "year" in group_col.lower():
                converted = pd.to_datetime(local[group_col], errors="coerce")
                if converted.notna().any():
                    local[group_col] = converted.dt.strftime("%Y-%m-%d")
        grouped = local.groupby(group_col, dropna=False)[valid_metrics]
        if agg == "mean":
            result = grouped.mean().reset_index()
        elif agg == "median":
            result = grouped.median().reset_index()
        elif agg == "min":
            result = grouped.min().reset_index()
        elif agg == "max":
            result = grouped.max().reset_index()
        elif agg == "count":
            result = grouped.count().reset_index()
        else:
            result = grouped.sum().reset_index()
        return result

    def available_chart_types(self, analysis_mode: str) -> List[str]:
        if analysis_mode == "dashboard":
            return ["column", "line", "bar", "box", "heatmap", "pie", "donut"]
        if analysis_mode == "multivariate":
            return ["column", "bar", "line", "area", "scatter", "box", "heatmap"]
        return ["column", "bar", "line", "area", "histogram", "box", "pie", "donut"]

    def create_chart(
        self,
        df: pd.DataFrame,
        chart_type: str,
        metrics: List[str],
        category_col: Optional[str] = None,
        date_col: Optional[str] = None,
        agg: str = "sum",
    ):
        if df.empty or not metrics:
            return None
        valid_metrics = [metric for metric in metrics if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric])]
        if not valid_metrics:
            return None

        group_col = date_col or category_col
        data = self.aggregate_metrics(df, valid_metrics, group_col=group_col, agg=agg)
        title_suffix = f" by {group_col}" if group_col else ""

        if chart_type == "heatmap":
            corr = df[valid_metrics].corr(numeric_only=True)
            if corr.empty:
                return None
            return px.imshow(corr, text_auto=".2f", aspect="auto", title="Metric Correlation Heatmap")

        if chart_type == "histogram":
            return px.histogram(df, x=valid_metrics[0], title=f"Distribution of {valid_metrics[0]}")

        if chart_type in ["pie", "donut"]:
            if not group_col:
                return None
            pie_metric = valid_metrics[0]
            fig = px.pie(data, names=group_col, values=pie_metric, title=f"{pie_metric}{title_suffix}")
            if chart_type == "donut":
                fig.update_traces(hole=0.45)
            return fig

        if chart_type == "box":
            box_x = category_col if category_col in df.columns else None
            return px.box(df, x=box_x, y=valid_metrics[0], points="outliers", title=f"Box Plot of {valid_metrics[0]}")

        if chart_type == "scatter":
            if len(valid_metrics) < 2:
                return None
            color_col = category_col if category_col in df.columns else None
            return px.scatter(
                df,
                x=valid_metrics[0],
                y=valid_metrics[1],
                color=color_col,
                title=f"{valid_metrics[0]} vs {valid_metrics[1]}",
            )

        plot_df = data if group_col else data.reset_index(drop=True)
        if group_col and group_col not in plot_df.columns:
            return None
        if len(valid_metrics) > 1 and group_col:
            melted = plot_df.melt(id_vars=[group_col], value_vars=valid_metrics, var_name="metric", value_name="value")
            if chart_type == "line":
                return px.line(melted, x=group_col, y="value", color="metric", markers=True, title=f"Metric Trend{title_suffix}")
            if chart_type == "area":
                return px.area(melted, x=group_col, y="value", color="metric", title=f"Metric Area View{title_suffix}")
            orientation = "h" if chart_type == "bar" else "v"
            return px.bar(
                melted,
                x="value" if orientation == "h" else group_col,
                y=group_col if orientation == "h" else "value",
                color="metric",
                barmode="group",
                orientation=orientation,
                title=f"Metric Comparison{title_suffix}",
            )

        metric = valid_metrics[0]
        if chart_type == "line":
            if group_col:
                return px.line(plot_df, x=group_col, y=metric, markers=True, title=f"{metric}{title_suffix}")
            return px.line(plot_df, y=metric, title=f"{metric} Trend")
        if chart_type == "area":
            if group_col:
                return px.area(plot_df, x=group_col, y=metric, title=f"{metric}{title_suffix}")
            return px.area(plot_df, y=metric, title=f"{metric} Area View")
        if chart_type == "bar":
            if not group_col:
                return None
            return px.bar(plot_df, x=metric, y=group_col, orientation="h", title=f"{metric}{title_suffix}")
        if chart_type == "column":
            if not group_col:
                return None
            return px.bar(plot_df, x=group_col, y=metric, title=f"{metric}{title_suffix}")
        return None

    def dashboard_figures(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        category_col: Optional[str] = None,
        date_col: Optional[str] = None,
        chart_types: Optional[List[str]] = None,
        agg: str = "sum",
    ) -> List:
        figures = []
        for chart_type in chart_types or ["column", "line", "heatmap"]:
            fig = self.create_chart(
                df=df,
                chart_type=chart_type,
                metrics=metrics,
                category_col=category_col,
                date_col=date_col,
                agg=agg,
            )
            if fig is not None:
                figures.append(fig)
        return figures

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
