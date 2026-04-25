from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def expand_to_15_minutes(df: pd.DataFrame, hour_column: str = "Hour") -> pd.DataFrame:
    """Duplicate each hourly row 4x and assign minute marks 00, 15, 30, 45."""
    if hour_column not in df.columns:
        raise ValueError(f"Column '{hour_column}' was not found in the input dataset")

    expanded = df.loc[df.index.repeat(4)].copy().reset_index(drop=True)
    expanded["Minute"] = np.tile([0, 15, 30, 45], len(df))

    # Keep a readable time string even if source data uses non-standard hour values (e.g., 25).
    hour_as_int = pd.to_numeric(expanded[hour_column], errors="coerce").fillna(0).astype(int)
    expanded["Time"] = hour_as_int.map(lambda h: f"{h:02d}") + ":" + expanded["Minute"].map(lambda m: f"{m:02d}")

    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand hourly CSV rows to 15-minute intervals (00, 15, 30, 45)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/denys/PycharmProjects/ds_demo/ds_project_demo/datasets_v4/dataset.csv"),
        help="Path to input hourly CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/denys/PycharmProjects/ds_demo/ds_project_demo/datasets_v5/dataset_15min.csv"),
        help="Path to output 15-minute CSV",
    )
    parser.add_argument(
        "--hour-column",
        default="Hour",
        help="Name of the hour column in input data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    expanded = expand_to_15_minutes(source, hour_column=args.hour_column)
    expanded.to_csv("expanded.csv", index=False)

    print(f"Input rows: {len(source)}")
    print(f"Output rows: {len(expanded)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

