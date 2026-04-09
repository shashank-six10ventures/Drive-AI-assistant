from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class AnalyticsEngine:
    def humanize_label(self, value: str) -> str:
        raw = str(value).replace("_", " ").strip()
        words = []
        acronyms = {"asin": "ASIN", "roi": "ROI", "fba": "FBA", "cog": "CoG", "sku": "SKU", "qty": "Qty"}
        for token in raw.split():
            lowered = token.lower()
            words.append(acronyms.get(lowered, token.capitalize()))
        return " ".join(words)

    def infer_dataset_type(self, df: pd.DataFrame) -> str:
        columns = " ".join([str(col).lower() for col in df.columns])
        if any(token in columns for token in ["asin", "fba", "referral fee", "marketplace", "product sales"]):
            return "amazon sales"
        if any(token in columns for token in ["revenue", "sales", "profit", "margin"]):
            return "sales and finance"
        if any(token in columns for token in ["campaign", "impressions", "clicks", "ctr", "spend"]):
            return "marketing"
        if any(token in columns for token in ["inventory", "stock", "supplier", "warehouse"]):
            return "inventory and operations"
        return "general business"

    def metric_definition(self, column_name: str) -> str:
        lower = str(column_name).lower()
        if "sales" in lower and "product" in lower:
            return "Revenue attributed directly to product sales before some downstream adjustments."
        if lower == "sales" or "revenue" in lower:
            return "Top-line money generated in the selected period or grouping."
        if "profit" in lower:
            return "Money left after costs and fees for the selected grouping."
        if "margin" in lower:
            return "Profit as a percentage of sales, useful for judging efficiency."
        if "roi" in lower:
            return "Return on investment, showing how much value was created relative to spend."
        if "fee" in lower:
            return "Marketplace or fulfillment cost charged against the sale."
        if "tax" in lower:
            return "Tax amount associated with the sale or transaction."
        if "refund" in lower:
            return "Money returned to customers or units reversed."
        if "qty" in lower or "units" in lower:
            return "Quantity sold or processed."
        if "cost" in lower or lower == "cog":
            return "Direct product or operating cost attached to the transaction."
        if "spend" in lower:
            return "Outgoing marketing or operating investment."
        if "click" in lower or "impression" in lower:
            return "Traffic or visibility metric used to judge funnel performance."
        return "Numeric business metric available for grouping, comparison, and trend analysis."

    def metric_definitions(self, df: pd.DataFrame, limit: int = 10) -> List[Dict]:
        profile = self.dataset_profile(df)
        ordered = profile["metric_suggestions"] + [col for col in profile["numeric_columns"] if col not in profile["metric_suggestions"]]
        return [
            {
                "metric": metric,
                "label": self.humanize_label(metric),
                "definition": self.metric_definition(metric),
            }
            for metric in ordered[:limit]
        ]

    def business_grain_options(self, df: pd.DataFrame) -> List[str]:
        profile = self.dataset_profile(df)
        options = []
        for col in profile["datetime_columns"]:
            options.append(col)
        for col in profile["categorical_columns"]:
            unique_count = df[col].nunique(dropna=True)
            if 1 < unique_count <= 50:
                options.append(col)
        return list(dict.fromkeys(options))

    def chart_title(self, chart_type: str, metrics: List[str], group_col: Optional[str], agg: str = "sum") -> str:
        metric_labels = ", ".join([self.humanize_label(metric) for metric in metrics[:2]])
        group_label = self.humanize_label(group_col) if group_col else "overall dataset"
        agg_label = {
            "sum": "total",
            "mean": "average",
            "median": "median",
            "min": "minimum",
            "max": "maximum",
            "count": "count",
        }.get(agg, agg)
        chart_labels = {
            "line": "trend",
            "area": "area trend",
            "column": "comparison",
            "bar": "ranked comparison",
            "box": "distribution",
            "scatter": "relationship",
            "heatmap": "correlation",
            "histogram": "distribution",
            "pie": "mix",
            "donut": "mix",
        }
        headline = f"{metric_labels} {chart_labels.get(chart_type, 'analysis')}"
        subtitle = f"Showing {agg_label} values by {group_label}" if group_col else f"Showing {agg_label} values for the full dataset"
        return f"{headline}<br><sup>{subtitle}</sup>"

    def metric_format_kind(self, column_name: str) -> str:
        lower = str(column_name).lower()
        if any(token in lower for token in ["margin", "roi", "ctr", "rate", "percent", "%"]):
            return "percent"
        if any(token in lower for token in ["sales", "revenue", "profit", "price", "cost", "cog", "fee", "tax", "spend", "amount", "income", "refund"]):
            return "currency"
        if any(token in lower for token in ["qty", "unit", "count", "orders", "click", "impression", "lead"]):
            return "count"
        return "number"

    def format_metric_value(self, column_name: str, value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        kind = self.metric_format_kind(column_name)
        numeric_value = float(value)
        if kind == "currency":
            return f"${numeric_value:,.2f}"
        if kind == "percent":
            if abs(numeric_value) <= 1:
                return f"{numeric_value * 100:,.2f}%"
            return f"{numeric_value:,.2f}%"
        if kind == "count":
            return f"{numeric_value:,.0f}"
        return f"{numeric_value:,.2f}"

    def apply_axis_format(self, fig, metric_name: str, axis: str = "y"):
        if fig is None:
            return fig
        kind = self.metric_format_kind(metric_name)
        if axis == "y":
            if kind == "currency":
                fig.update_yaxes(tickprefix="$")
            elif kind == "percent":
                fig.update_yaxes(ticksuffix="%")
        else:
            if kind == "currency":
                fig.update_xaxes(tickprefix="$")
            elif kind == "percent":
                fig.update_xaxes(ticksuffix="%")
        return fig

    def explain_chart_choice(
        self,
        chart_type: str,
        metrics: List[str],
        category_col: Optional[str] = None,
        date_col: Optional[str] = None,
        agg: str = "sum",
    ) -> str:
        metric_label = ", ".join([self.humanize_label(metric) for metric in metrics[:2]])
        if chart_type == "line":
            return f"Chosen because {metric_label} is easier to read as a trend over time and spot changes in direction."
        if chart_type == "area":
            return f"Chosen because {metric_label} benefits from seeing cumulative movement over time."
        if chart_type == "column":
            return f"Chosen because it compares {metric_label} clearly across grouped business buckets like {self.humanize_label(category_col or date_col or 'the selected dimension')}."
        if chart_type == "bar":
            return f"Chosen because a ranked horizontal comparison makes it easier to scan the biggest and weakest groups for {metric_label}."
        if chart_type == "heatmap":
            return f"Chosen because it shows how strongly the selected metrics move together, which is useful for correlation checks."
        if chart_type == "box":
            return f"Chosen because it highlights spread and outliers in {metric_label}."
        if chart_type == "scatter":
            return f"Chosen because it helps check the relationship between {metric_label} values."
        if chart_type in ["pie", "donut"]:
            return f"Chosen because it shows the share mix of {metric_label} across the selected groups."
        if chart_type == "histogram":
            return f"Chosen because it shows how {metric_label} is distributed across the dataset."
        return f"Chosen to summarize {metric_label} using `{agg}` aggregation."

    def load_dataset(self, local_path: str) -> pd.DataFrame:
        path = Path(local_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
            return self.clean_dataframe(df)
        if suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
            return self.clean_dataframe(df)
        raise ValueError(f"Unsupported dataset type for analytics: {suffix}")

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        local = df.copy()
        local = local.replace(
            {
                "None": pd.NA,
                "none": pd.NA,
                "null": pd.NA,
                "NULL": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "": pd.NA,
            }
        )
        local = local.dropna(how="all")
        for col in local.columns:
            if local[col].dtype == "object":
                numeric = pd.to_numeric(local[col], errors="coerce")
                if numeric.notna().mean() > 0.8:
                    local[col] = numeric
        return local.reset_index(drop=True)

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
            return {"count": 0, "rows": [], "method": "", "bounds": {}}
        series = df[value_col].dropna()
        if len(series) < 4:
            return {
                "count": 0,
                "rows": [],
                "method": "IQR anomaly detection skipped because there are fewer than 4 numeric observations.",
                "bounds": {},
            }
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[value_col] < lower) | (df[value_col] > upper)].copy()
        if date_col and date_col in outliers.columns:
            outliers[date_col] = pd.to_datetime(outliers[date_col], errors="coerce")
        return {
            "count": int(len(outliers)),
            "rows": outliers.head(20).to_dict(orient="records"),
            "method": (
                f"IQR method on `{value_col}`: values below {lower:,.2f} or above {upper:,.2f} "
                f"are flagged as anomalies."
            ),
            "bounds": {"lower": float(lower), "upper": float(upper), "q1": float(q1), "q3": float(q3)},
        }

    def detect_grouped_anomalies(self, df: pd.DataFrame, value_col: str, group_col: str, agg: str = "sum") -> Dict:
        if not group_col or group_col not in df.columns:
            return {"count": 0, "rows": [], "method": "", "bounds": {}, "group_col": group_col}
        grouped = self.aggregate_metrics(df, [value_col], group_col=group_col, agg=agg)
        if grouped.empty or value_col not in grouped.columns:
            return {"count": 0, "rows": [], "method": "", "bounds": {}, "group_col": group_col}
        series = grouped[value_col].dropna()
        if len(series) < 4:
            return {
                "count": 0,
                "rows": [],
                "method": f"Grouped anomaly detection skipped for `{group_col}` because there are fewer than 4 aggregated observations.",
                "bounds": {},
                "group_col": group_col,
            }
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = grouped[(grouped[value_col] < lower) | (grouped[value_col] > upper)].copy()
        if not flagged.empty:
            flagged = flagged.sort_values(value_col, ascending=False)
        return {
            "count": int(len(flagged)),
            "rows": flagged.head(25).to_dict(orient="records"),
            "method": (
                f"IQR method on grouped `{self.humanize_label(value_col)}` by `{self.humanize_label(group_col)}` "
                f"using `{agg}` aggregation. Groups below {lower:,.2f} or above {upper:,.2f} are flagged."
            ),
            "bounds": {"lower": float(lower), "upper": float(upper), "q1": float(q1), "q3": float(q3)},
            "group_col": group_col,
            "grouped_table": grouped.to_dict(orient="records"),
        }

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

    def best_grouping_dimension(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        profile = self.dataset_profile(df)
        time_dim = profile["datetime_columns"][0] if profile["datetime_columns"] else None
        category_dim = None
        priority_tokens = ["asin", "category", "product", "sku", "campaign", "channel", "region", "brand"]
        for token in priority_tokens:
            for col in profile["categorical_columns"]:
                if token in col.lower():
                    unique_count = df[col].nunique(dropna=True)
                    if 1 < unique_count <= 50:
                        category_dim = col
                        break
            if category_dim:
                break
        if not category_dim:
            for col in profile["categorical_columns"]:
                unique_count = df[col].nunique(dropna=True)
                if 1 < unique_count <= 25:
                    category_dim = col
                    break
        return {"time": time_dim, "category": category_dim}

    def find_column_by_query(
        self,
        df: pd.DataFrame,
        query: str,
        preferred_types: Optional[List[str]] = None,
    ) -> Optional[str]:
        profile = self.dataset_profile(df)
        query_lower = query.lower()
        preferred_types = preferred_types or ["any"]
        candidate_columns: List[str] = []
        if "numeric" in preferred_types:
            candidate_columns.extend(profile["numeric_columns"])
        if "categorical" in preferred_types:
            candidate_columns.extend(profile["categorical_columns"])
        if "datetime" in preferred_types:
            candidate_columns.extend(profile["datetime_columns"])
        if "any" in preferred_types or not candidate_columns:
            candidate_columns = [str(col) for col in df.columns]

        ranked = []
        query_tokens = [token for token in re.split(r"[^a-zA-Z0-9]+", query_lower) if token]
        for col in candidate_columns:
            col_lower = col.lower()
            score = 0
            for token in query_tokens:
                if token and token in col_lower:
                    score += 2
            normalized_col = col_lower.replace("_", " ")
            if normalized_col in query_lower:
                score += 3
            if score > 0:
                ranked.append((score, col))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1] if ranked else None

    def resolve_query_dimensions(self, df: pd.DataFrame, query: str) -> Dict[str, Optional[str]]:
        profile = self.dataset_profile(df)
        dims = self.best_grouping_dimension(df)
        metric = self.find_column_by_query(df, query, preferred_types=["numeric"])
        if not metric:
            metric = self.find_best_column(df, ["revenue", "sales", "profit", "price", "amount", "roi", "margin"])
        category = self.find_column_by_query(df, query, preferred_types=["categorical"])
        wise_match = re.search(r"([a-zA-Z0-9_ ]+?)\s+wise", query.lower())
        if wise_match:
            requested = wise_match.group(1).strip()
            for col in profile["categorical_columns"] + profile["datetime_columns"]:
                if requested and requested in col.lower():
                    category = col
                    break
        group_match = re.search(r"group\s+by\s+([a-zA-Z0-9_ ]+)", query.lower())
        if group_match:
            requested = group_match.group(1).strip()
            for col in profile["categorical_columns"] + profile["datetime_columns"]:
                if requested and requested in col.lower():
                    category = col
                    break
        time_dim = self.find_column_by_query(df, query, preferred_types=["datetime"])
        if not category:
            category = dims["category"]
        if not time_dim:
            time_dim = dims["time"]
        return {
            "metric": metric,
            "category": category,
            "time": time_dim,
            "profile": profile,
        }

    def answer_dataset_question(self, df: pd.DataFrame, query: str) -> Dict:
        query_lower = query.lower()
        resolved = self.resolve_query_dimensions(df, query)
        metric = resolved["metric"]
        category = resolved["category"]
        time_dim = resolved["time"]
        if not metric:
            return {
                "text": "I couldn't identify a numeric metric in the current dataset for that question.",
                "table": pd.DataFrame(),
                "chart": None,
            }

        top_n_match = re.search(r"top\s+(\d+)", query_lower)
        top_n = int(top_n_match.group(1)) if top_n_match else 5

        if "trend" in query_lower or "over time" in query_lower or "daily" in query_lower or "monthly" in query_lower:
            if not time_dim:
                return {
                    "text": f"I found metric `{metric}`, but this dataset does not have a clear time column for trend analysis.",
                    "table": pd.DataFrame(),
                    "chart": None,
                }
            trend_table = self.aggregate_metrics(df, [metric], group_col=time_dim, agg="sum")
            trend_chart = self.create_chart(trend_table, "line", [metric], category_col=time_dim)
            return {
                "text": f"Showing the trend of `{metric}` over `{time_dim}` for the current dataset.",
                "table": trend_table,
                "chart": trend_chart,
            }

        if "wise" in query_lower or "group by" in query_lower or "by " in query_lower:
            group_col = category or time_dim
            if group_col:
                ranked_table = self.aggregate_metrics(df, [metric], group_col=group_col, agg="sum")
                if ranked_table.empty:
                    return {
                        "text": f"I found metric `{metric}`, but couldn't group the current dataset cleanly by `{group_col}`.",
                        "table": pd.DataFrame(),
                        "chart": None,
                    }
                ascending = any(token in query_lower for token in ["lowest", "bottom", "worst", "ascending"])
                ranked_table = ranked_table.sort_values(metric, ascending=ascending)
                if "top" in query_lower or "highest" in query_lower or "lowest" in query_lower or "bottom" in query_lower:
                    ranked_table = ranked_table.head(top_n)
                chart_type = "bar" if group_col == category else "line"
                return {
                    "text": f"Showing `{metric}` grouped by `{group_col}` for the current dataset.",
                    "table": ranked_table,
                    "chart": self.create_chart(ranked_table, chart_type, [metric], category_col=group_col),
                }

        if any(token in query_lower for token in ["top", "highest", "lowest", "bottom", "rank", "best", "worst"]):
            group_col = category or time_dim
            if not group_col:
                total_value = float(df[metric].sum())
                return {
                    "text": f"The dataset does not have a clear grouping dimension, so the total `{metric}` is {total_value:,.2f}.",
                    "table": pd.DataFrame(),
                    "chart": None,
                }
            ranked_table = self.aggregate_metrics(df, [metric], group_col=group_col, agg="sum")
            ascending = any(token in query_lower for token in ["lowest", "bottom", "worst"])
            ranked_table = ranked_table.sort_values(metric, ascending=ascending).head(top_n)
            chart_type = "bar" if group_col == category else "line"
            ranked_chart = self.create_chart(ranked_table, chart_type, [metric], category_col=group_col)
            descriptor = "lowest" if ascending else "top"
            return {
                "text": f"Showing the {descriptor} {len(ranked_table)} `{group_col}` values ranked by `{metric}` in the current dataset.",
                "table": ranked_table,
                "chart": ranked_chart,
            }

        if any(token in query_lower for token in ["average", "mean", "sum", "total", "median", "min", "max"]):
            agg = "sum"
            if "average" in query_lower or "mean" in query_lower:
                agg = "mean"
            elif "median" in query_lower:
                agg = "median"
            elif "min" in query_lower or "lowest" in query_lower:
                agg = "min"
            elif "max" in query_lower or "highest" in query_lower:
                agg = "max"
            summary_table = self.aggregate_metrics(df, [metric], group_col=category or time_dim, agg=agg)
            return {
                "text": f"Showing `{agg}` analysis for `{metric}` in the current dataset.",
                "table": summary_table,
                "chart": self.create_chart(summary_table, "column", [metric], category_col=category or time_dim),
            }

        return {
            "text": (
                f"I recognized metric `{metric}` for the current dataset. "
                f"Try asking for top groups, trend over time, totals, averages, or rankings."
            ),
            "table": pd.DataFrame(),
            "chart": None,
        }

    def explain_table(self, df: pd.DataFrame) -> str:
        profile = self.dataset_profile(df)
        lines = [
            f"This table has {profile['row_count']} rows and {profile['column_count']} columns.",
        ]
        if profile["datetime_columns"]:
            lines.append(f"Time fields detected: {', '.join(profile['datetime_columns'][:3])}.")
        if profile["metric_suggestions"]:
            lines.append(f"Suggested metrics for analysis: {', '.join(profile['metric_suggestions'][:5])}.")
        if profile["categorical_columns"]:
            lines.append(f"Useful grouping dimensions: {', '.join(profile['categorical_columns'][:5])}.")
        return " ".join(lines)

    def default_dashboard_bundle(self, df: pd.DataFrame) -> Dict:
        profile = self.dataset_profile(df)
        dims = self.best_grouping_dimension(df)
        dataset_type = self.infer_dataset_type(df)
        metrics = profile["metric_suggestions"][: min(3, len(profile["metric_suggestions"]))]
        if not metrics:
            metrics = profile["numeric_columns"][: min(3, len(profile["numeric_columns"]))]
        chart_types = ["column", "line", "heatmap"] if dims["time"] else ["column", "bar", "heatmap"]
        if dataset_type == "amazon sales":
            chart_types = ["line", "column", "heatmap"] if dims["time"] else ["bar", "column", "heatmap"]
        elif dataset_type == "marketing":
            chart_types = ["line", "bar", "heatmap"] if dims["time"] else ["bar", "pie", "heatmap"]
        elif dataset_type == "inventory and operations":
            chart_types = ["column", "bar", "heatmap"]
        panels = self.dashboard_panels(
            df=df,
            metrics=metrics[:2] if metrics else [],
            category_col=dims["category"],
            date_col=dims["time"],
            chart_types=chart_types,
            agg="sum",
        )
        chart_reasons = [
            {
                "chart_type": chart_type,
                "reason": self.explain_chart_choice(chart_type, metrics[:2], dims["category"], dims["time"], "sum"),
            }
            for chart_type in chart_types
        ]
        grain_options = self.business_grain_options(df)
        anomaly_grain = dims["category"] or dims["time"] or (grain_options[0] if grain_options else None)
        grouped_anomalies = self.detect_grouped_anomalies(df, metrics[0], anomaly_grain, agg="sum") if metrics and anomaly_grain else {"count": 0, "rows": [], "method": "", "bounds": {}}
        narrative = (
            f"Recommended dashboard for a {dataset_type} dataset. "
            f"Lead with `{self.humanize_label(metrics[0])}`"
            + (f", trend it over `{self.humanize_label(dims['time'])}`" if dims["time"] else "")
            + (f", and compare it by `{self.humanize_label(dims['category'])}`." if dims["category"] else ".")
        )
        return {
            "profile": profile,
            "dimensions": dims,
            "dataset_type": dataset_type,
            "metrics": metrics,
            "figures": [panel["figure"] for panel in panels],
            "panels": panels,
            "chart_reasons": chart_reasons,
            "kpis": self.metric_kpis(df, metrics),
            "metric_definitions": self.metric_definitions(df),
            "grain_options": grain_options,
            "grouped_anomalies": grouped_anomalies,
            "narrative": narrative,
        }

    def benchmark_cards(
        self,
        df: pd.DataFrame,
        metric_col: Optional[str],
        category_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> List[Dict]:
        if not metric_col or metric_col not in df.columns or not pd.api.types.is_numeric_dtype(df[metric_col]):
            return []

        cards: List[Dict] = []
        if category_col and category_col in df.columns:
            grouped = self.aggregate_metrics(df, [metric_col], group_col=category_col, agg="sum")
            if not grouped.empty and metric_col in grouped.columns:
                ranked = grouped.sort_values(metric_col, ascending=False).reset_index(drop=True)
                best_row = ranked.iloc[0]
                weakest_row = ranked.iloc[-1]
                cards.append(
                    {
                        "label": f"Best {self.humanize_label(category_col)}",
                        "value": str(best_row[category_col]),
                        "metric": metric_col,
                        "formatted": self.format_metric_value(metric_col, best_row[metric_col]),
                        "detail": f"Highest total {self.humanize_label(metric_col)}",
                    }
                )
                cards.append(
                    {
                        "label": f"Weakest {self.humanize_label(category_col)}",
                        "value": str(weakest_row[category_col]),
                        "metric": metric_col,
                        "formatted": self.format_metric_value(metric_col, weakest_row[metric_col]),
                        "detail": f"Lowest total {self.humanize_label(metric_col)}",
                    }
                )

        if date_col and date_col in df.columns:
            trend = self.aggregate_metrics(df, [metric_col], group_col=date_col, agg="sum")
            if not trend.empty and metric_col in trend.columns:
                local = trend.copy()
                local[date_col] = pd.to_datetime(local[date_col], errors="coerce")
                local = local.dropna(subset=[date_col]).sort_values(date_col)
                if len(local) >= 2:
                    local["_change"] = local[metric_col].diff()
                    fastest = local.loc[local["_change"].idxmax()] if local["_change"].notna().any() else None
                    if fastest is not None and pd.notna(fastest["_change"]):
                        cards.append(
                            {
                                "label": f"Fastest Growth {self.humanize_label(date_col)}",
                                "value": str(fastest[date_col].date()),
                                "metric": metric_col,
                                "formatted": self.format_metric_value(metric_col, fastest["_change"]),
                                "detail": f"Largest period-over-period increase in {self.humanize_label(metric_col)}",
                            }
                        )
                best_day = local.loc[local[metric_col].idxmax()] if not local.empty else None
                if best_day is not None:
                    cards.append(
                        {
                            "label": f"Best {self.humanize_label(date_col)}",
                            "value": str(best_day[date_col].date()),
                            "metric": metric_col,
                            "formatted": self.format_metric_value(metric_col, best_day[metric_col]),
                            "detail": f"Highest total {self.humanize_label(metric_col)}",
                        }
                    )
        return cards[:4]

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
            return px.imshow(corr, text_auto=".2f", aspect="auto", title=self.chart_title(chart_type, valid_metrics, group_col, agg))

        if chart_type == "histogram":
            return px.histogram(df, x=valid_metrics[0], title=self.chart_title(chart_type, valid_metrics, group_col, agg))

        if chart_type in ["pie", "donut"]:
            if not group_col:
                return None
            pie_metric = valid_metrics[0]
            fig = px.pie(data, names=group_col, values=pie_metric, title=self.chart_title(chart_type, [pie_metric], group_col, agg))
            if chart_type == "donut":
                fig.update_traces(hole=0.45)
            return fig

        if chart_type == "box":
            box_x = category_col if category_col in df.columns else None
            return px.box(df, x=box_x, y=valid_metrics[0], points="outliers", title=self.chart_title(chart_type, valid_metrics, box_x, agg))

        if chart_type == "scatter":
            if len(valid_metrics) < 2:
                return None
            color_col = category_col if category_col in df.columns else None
            return px.scatter(
                df,
                x=valid_metrics[0],
                y=valid_metrics[1],
                color=color_col,
                title=self.chart_title(chart_type, valid_metrics, color_col, agg),
            )

        plot_df = data if group_col else data.reset_index(drop=True)
        if group_col and group_col not in plot_df.columns:
            return None
        if len(valid_metrics) > 1 and group_col:
            melted = plot_df.melt(id_vars=[group_col], value_vars=valid_metrics, var_name="metric", value_name="value")
            if chart_type == "line":
                fig = px.line(melted, x=group_col, y="value", color="metric", markers=True, title=self.chart_title(chart_type, valid_metrics, group_col, agg))
                return self.apply_axis_format(fig, valid_metrics[0], axis="y")
            if chart_type == "area":
                fig = px.area(melted, x=group_col, y="value", color="metric", title=self.chart_title(chart_type, valid_metrics, group_col, agg))
                return self.apply_axis_format(fig, valid_metrics[0], axis="y")
            orientation = "h" if chart_type == "bar" else "v"
            fig = px.bar(
                melted,
                x="value" if orientation == "h" else group_col,
                y=group_col if orientation == "h" else "value",
                color="metric",
                barmode="group",
                orientation=orientation,
                title=self.chart_title(chart_type, valid_metrics, group_col, agg),
            )
            return self.apply_axis_format(fig, valid_metrics[0], axis="x" if orientation == "h" else "y")

        metric = valid_metrics[0]
        if chart_type == "line":
            if group_col:
                fig = px.line(plot_df, x=group_col, y=metric, markers=True, title=self.chart_title(chart_type, [metric], group_col, agg))
                return self.apply_axis_format(fig, metric, axis="y")
            fig = px.line(plot_df, y=metric, title=self.chart_title(chart_type, [metric], group_col, agg))
            return self.apply_axis_format(fig, metric, axis="y")
        if chart_type == "area":
            if group_col:
                fig = px.area(plot_df, x=group_col, y=metric, title=self.chart_title(chart_type, [metric], group_col, agg))
                return self.apply_axis_format(fig, metric, axis="y")
            fig = px.area(plot_df, y=metric, title=self.chart_title(chart_type, [metric], group_col, agg))
            return self.apply_axis_format(fig, metric, axis="y")
        if chart_type == "bar":
            if not group_col:
                return None
            fig = px.bar(plot_df, x=metric, y=group_col, orientation="h", title=self.chart_title(chart_type, [metric], group_col, agg))
            return self.apply_axis_format(fig, metric, axis="x")
        if chart_type == "column":
            if not group_col:
                return None
            fig = px.bar(plot_df, x=group_col, y=metric, title=self.chart_title(chart_type, [metric], group_col, agg))
            return self.apply_axis_format(fig, metric, axis="y")
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
        for panel in self.dashboard_panels(df, metrics, category_col, date_col, chart_types, agg):
            figures.append(panel["figure"])
        return figures

    def chart_source_data(
        self,
        df: pd.DataFrame,
        chart_type: str,
        metrics: List[str],
        category_col: Optional[str] = None,
        date_col: Optional[str] = None,
        agg: str = "sum",
    ) -> pd.DataFrame:
        valid_metrics = [metric for metric in metrics if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric])]
        if not valid_metrics:
            return pd.DataFrame()
        group_col = date_col or category_col
        if chart_type == "heatmap":
            return df[valid_metrics].corr(numeric_only=True).reset_index()
        if chart_type in ["histogram", "box"]:
            source_cols = [valid_metrics[0]]
            if category_col and category_col in df.columns:
                source_cols = [category_col] + source_cols
            return df[source_cols].copy()
        if chart_type == "scatter":
            if len(valid_metrics) < 2:
                return pd.DataFrame()
            source_cols = valid_metrics[:2]
            if category_col and category_col in df.columns:
                source_cols.append(category_col)
            return df[source_cols].copy()
        return self.aggregate_metrics(df, valid_metrics, group_col=group_col, agg=agg)

    def dashboard_panels(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        category_col: Optional[str] = None,
        date_col: Optional[str] = None,
        chart_types: Optional[List[str]] = None,
        agg: str = "sum",
    ) -> List[Dict]:
        panels = []
        for chart_type in chart_types or ["column", "line", "heatmap"]:
            fig = self.create_chart(
                df=df,
                chart_type=chart_type,
                metrics=metrics,
                category_col=category_col,
                date_col=date_col,
                agg=agg,
            )
            if fig is None:
                continue
            panels.append(
                {
                    "chart_type": chart_type,
                    "figure": fig,
                    "data": self.chart_source_data(df, chart_type, metrics, category_col, date_col, agg),
                    "reason": self.explain_chart_choice(chart_type, metrics, category_col, date_col, agg),
                }
            )
        return panels

    def business_alerts(self, datasets: List[Dict], metric_hints: Optional[List[str]] = None) -> List[Dict]:
        alerts = []
        for item in datasets:
            df = item["df"]
            metric_col = self.find_best_column(df, metric_hints or ["revenue", "sales", "profit", "price"])
            if not metric_col:
                continue
            dims = self.best_grouping_dimension(df)
            anomaly_grain = dims["category"] or dims["time"]
            anomalies = (
                self.detect_grouped_anomalies(df, metric_col, anomaly_grain, agg="sum")
                if anomaly_grain
                else self.detect_anomalies(df, metric_col)
            )
            if anomalies["count"] > 0:
                alerts.append(
                    {
                        "file_name": item["file_name"],
                        "severity": "high" if anomalies["count"] >= 3 else "medium",
                        "message": (
                            f"{anomalies['count']} unusual `{metric_col}` values detected"
                            + (f" when grouped by `{anomaly_grain}`." if anomaly_grain else ".")
                        ),
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
        local["_trend_bucket"] = local[date_candidate].dt.strftime("%Y-%m-%d")
        grouped = local.groupby("_trend_bucket", as_index=False)[value_candidate].sum()
        fig = px.line(grouped, x="_trend_bucket", y=value_candidate, markers=True, title=self.chart_title("line", [value_candidate], date_candidate, "sum"))
        return self.apply_axis_format(fig, value_candidate, axis="y")

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
        fig = px.bar(grouped, x=cat, y=val, title=self.chart_title("column", [val], cat, "sum"))
        return self.apply_axis_format(fig, val, axis="y")

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
