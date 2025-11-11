#!/usr/bin/env python3
"""Profile a Parquet dataset and write JSON + Markdown summaries."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PERCENTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _sanitize(value):
    """Convert values to JSON-serialisable representations."""
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    return value


def load_dataframe(path: Path, sample_size: Optional[int], engine: str) -> pd.DataFrame:
    """Read Parquet using the requested engine and optionally sample rows."""
    if engine == "pyarrow":
        try:
            import pyarrow.parquet as pq  # type: ignore

            table = pq.read_table(path)
            df = table.to_pandas()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("pyarrow read failed (%s); falling back to pandas.read_parquet", exc)
            df = pd.read_parquet(path)
    else:
        df = pd.read_parquet(path)

    if sample_size is not None and sample_size > 0:
        df = df.head(sample_size)
    return df


def profile_column(
    series: pd.Series,
    value_count_limit: Optional[int],
    percentiles: Iterable[float],
) -> Dict[str, object]:
    """Return column-level statistics."""
    info: Dict[str, object] = {
        "dtype": str(series.dtype),
        "missing_count": int(series.isna().sum()),
        "missing_pct": float(series.isna().mean() * 100),
        "nunique": int(series.nunique(dropna=True)),
    }

    non_null = series.dropna()
    example_values = [str(v)[:120] for v in non_null.head(3).tolist()]
    info["example_values"] = example_values

    if pd.api.types.is_numeric_dtype(series):
        stats = non_null.describe(percentiles=percentiles).to_dict()
        info["numeric_stats"] = {k: _sanitize(v) for k, v in stats.items()}
    elif pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        info["temporal_stats"] = {
            "min": _sanitize(non_null.min()) if not non_null.empty else None,
            "max": _sanitize(non_null.max()) if not non_null.empty else None,
        }
    else:
        counts = non_null.astype(str).value_counts(dropna=False)
        if value_count_limit is not None and value_count_limit > 0:
            counts = counts.head(value_count_limit)
        info["top_values"] = [
            {"value": str(val), "count": int(count)} for val, count in counts.items()
        ]
        info["value_count_truncated"] = value_count_limit is not None and value_count_limit > 0 and len(counts) == value_count_limit
    return info


def dataframe_profile(
    df: pd.DataFrame,
    value_count_limit: Optional[int],
    percentiles: Iterable[float],
) -> Dict[str, object]:
    """Compute dataset-level metrics and per-column profiles."""
    profile: Dict[str, object] = {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "missing_total": int(df.isna().sum().sum()),
    }
    profile["approx_row_size_bytes"] = (
        float(profile["memory_usage_bytes"]) / max(profile["rows"], 1)
        if profile["rows"]
        else 0.0
    )

    profile["columns_profile"] = {
        col: profile_column(df[col], value_count_limit, percentiles) for col in df.columns
    }
    return profile


def build_markdown(
    dataset_path: Path,
    profile: Dict[str, object],
    head_df: pd.DataFrame,
    value_count_limit: Optional[int],
) -> str:
    lines: List[str] = []
    lines.append(f"# Dataset summary — {dataset_path.name}")
    lines.append("")
    lines.append(f"* Source: `{dataset_path}`")
    lines.append(f"* Rows (after optional sampling): {profile['rows']}")
    lines.append(f"* Columns: {len(profile['columns'])}")
    lines.append(f"* Duplicate rows: {profile['duplicate_rows']}")
    lines.append(f"* Missing values (total): {profile['missing_total']}")
    lines.append(f"* Memory usage (approx.): {profile['memory_usage_bytes']:,} bytes")
    lines.append(f"* Average row size: {profile['approx_row_size_bytes']:.2f} bytes")
    lines.append("")
    
    if not head_df.empty:
        lines.append("## Sample rows")
    
        try:
            lines.append(head_df.to_markdown(index=False))
        except Exception:  # pylint: disable=broad-exception-caught
            lines.append("```")
            lines.append(head_df.to_string(index=False))
            lines.append("```")
        lines.append("")

    # Column overview table
    lines.append("## Column overview")
    lines.append("| Column | dtype | Missing % | Unique | Examples |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for col in profile["columns"]:
        col_info = profile["columns_profile"][col]
        examples_raw = ", ".join(col_info.get("example_values", []))
        examples = examples_raw.replace("|", "\|")
        lines.append(
            f"| {col} | {col_info['dtype']} | {col_info['missing_pct']:.2f} | {col_info['nunique']} | {examples} |"
        )

    # Detail sections per dtype
    numeric_cols = [
        col for col, info in profile["columns_profile"].items() if "numeric_stats" in info
    ]
    if numeric_cols:
        lines.append("## Numeric column statistics")
        for col in numeric_cols:
            stats = profile["columns_profile"][col]["numeric_stats"]
            lines.append(f"### {col}")
            for key, val in stats.items():
                lines.append(f"- {key}: {val}")
            lines.append("")

    temporal_cols = [
        col for col, info in profile["columns_profile"].items() if "temporal_stats" in info
    ]
    if temporal_cols:
        lines.append("## Temporal columns")
        for col in temporal_cols:
            stats = profile["columns_profile"][col]["temporal_stats"]
            lines.append(f"- {col}: min={stats['min']}, max={stats['max']}")
        lines.append("")

    categorical_cols = [
        col
        for col, info in profile["columns_profile"].items()
        if "top_values" in info and info["top_values"]
    ]
    if categorical_cols:
        lines.append("## Top categorical values")
        note = (
            " (first ``{}`` values)".format(value_count_limit)
            if value_count_limit is not None and value_count_limit > 0
            else ""
        )
        lines.append(f"The counts below show the most frequent values{note}.")
        lines.append("")
        for col in categorical_cols:
            lines.append(f"### {col}")
            lines.append("| Value | Count |")
            lines.append("| --- | ---: |")
            for item in profile["columns_profile"][col]["top_values"]:
                value_str = str(item['value']).replace('|', '\|')
                if len(value_str) > 120:
                    value_str = value_str[:117] + "..."
                lines.append(f"| {value_str} | {item['count']} |")
            lines.append("")

    return "\n".join(lines)


def write_outputs(
    dataset_path: Path,
    profile: Dict[str, object],
    head_df: pd.DataFrame,
    output_dir: Path,
    value_count_limit: Optional[int],
) -> None:
    base_name = dataset_path.stem

    json_payload = {
        "dataset": {
            "path": str(dataset_path.resolve()),
            "rows": profile["rows"],
            "columns": profile["columns"],
            "duplicate_rows": profile["duplicate_rows"],
            "memory_usage_bytes": profile["memory_usage_bytes"],
            "approx_row_size_bytes": profile["approx_row_size_bytes"],
            "missing_total": profile["missing_total"],
        },
        "columns": profile["columns_profile"],
        "head": [_sanitize(row) for row in head_df.head().to_dict(orient="records")],
    }

    json_path = output_dir / f"{base_name}_profile.json"
    json_path.write_text(json.dumps(_sanitize(json_payload), indent=2))
    logger.info("Wrote JSON profile to %s", json_path)

    md_content = build_markdown(dataset_path, profile, head_df, value_count_limit)
    md_path = output_dir / f"{base_name}_summary.md"
    md_path.write_text(md_content)
    logger.info("Wrote Markdown summary to %s", md_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_path", type=Path, help="Path to the Parquet dataset")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of rows to load (default: all)",
    )
    parser.add_argument(
        "--head-rows",
        type=int,
        default=5,
        help="Number of sample rows to show in the Markdown report",
    )
    parser.add_argument(
        "--value-count-limit",
        type=int,
        default=20,
        help="Limit of categorical value counts to include (<=0 means include all)",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="*",
        default=DEFAULT_PERCENTILES,
        help="Percentiles for numeric summaries",
    )
    parser.add_argument(
        "--engine",
        choices=["pandas", "pyarrow"],
        default="pandas",
        help="Parquet reader to use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis_outputs"),
        help="Directory for JSON/Markdown outputs",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if not args.parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {args.parquet_path}")

    df = load_dataframe(args.parquet_path, args.sample_size, args.engine)

    value_count_limit = args.value_count_limit
    if value_count_limit is not None and value_count_limit <= 0:
        value_count_limit = None

    profile = dataframe_profile(df, value_count_limit, args.percentiles)

    head_rows = max(args.head_rows, 0)
    head_df = df.head(head_rows) if head_rows else pd.DataFrame()
    
    # cap the length of long string entries in head_df to 120 characters
    for col in head_df.select_dtypes(include=['object']).columns:
        head_df[col] = head_df[col].apply(lambda x: str(x)[:117] + "..." if pd.notnull(x) and len(str(x)) > 120 else x)
        

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.parquet_path, profile, head_df, args.output_dir, value_count_limit)


if __name__ == "__main__":
    main()
