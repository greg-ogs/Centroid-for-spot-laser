"""Plotting utilities for benchmark CSVs (times + per-image coordinates).

The class is constructed with an explicit CSV path and value column; the
constructor does no I/O beyond reading the file. Plotting methods take an
explicit output path, write the figure, and close it.

Run from the project root:
    python -m source.visualization.times_graph results-c.csv
    python -m source.visualization.times_graph results-c.csv --out-dir results/plots
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

DEFAULT_VALUE_COL = "Time"
ALGORITHM_COLUMNS_TIMES = ["SLIC Time", "Felzenszwalb Time", "Quickshift Time",
                           "FBM Time", "CCL Time"]
ALGORITHM_COLUMNS_X = ["SLIC X", "Felzenszwalb X", "Quickshift X",
                       "FBM X", "CCL X"]
ALGORITHM_COLUMNS_Y = ["SLIC Y", "Felzenszwalb Y", "Quickshift Y",
                       "FBM Y", "CCL Y"]
ALGORITHM_COLUMNS_ERROR = ["SLIC", "Felzenszwalb", "Quickshift", "CCL", "FBM"]


class TimesGraph:

    def __init__(self, csv_path, *,
                 value_col: str = DEFAULT_VALUE_COL,
                 id_col=None,
                 value_columns: Sequence[str] | None = None):
        self.csv_path = str(csv_path)
        self.value_col = value_col

        df = pd.read_csv(self.csv_path)

        if value_columns is None:
            if value_col.lower() == "time":
                value_columns = [c for c in df.columns
                                 if isinstance(c, str) and c.endswith(" Time")]
            elif value_col.lower() == "x":
                value_columns = [c for c in ALGORITHM_COLUMNS_X if c in df.columns]
            elif value_col.lower() == "y":
                value_columns = [c for c in ALGORITHM_COLUMNS_Y if c in df.columns]
            else:
                value_columns = [c for c in ALGORITHM_COLUMNS_ERROR if c in df.columns]
        if not value_columns:
            raise ValueError(
                f"No value columns to melt in {self.csv_path}. "
                f"Pass value_columns=... explicitly."
            )

        if id_col is None:
            id_col = "ID" if "ID" in df.columns else [c for c in df.columns
                                                     if c not in value_columns]
        elif isinstance(id_col, str):
            id_col = [id_col]

        self.df_long = df.melt(id_vars=list(id_col), value_vars=list(value_columns),
                               var_name="Algorithm", value_name=value_col)

    # -------- pure helpers --------------------------------------------------

    @staticmethod
    def melt(csv_path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        time_columns = [c for c in df.columns
                        if isinstance(c, str) and c.endswith(" Time")]
        if not time_columns:
            raise ValueError(f"{csv_path}: no time columns to melt")
        id_vars = [c for c in df.columns if c not in time_columns]
        return df.melt(id_vars=id_vars, value_vars=time_columns,
                       var_name="Algorithm", value_name="Time")

    # -------- plotting (each method writes one file then closes) ----------

    def boxplot(self, output_path: str = "boxplot.png") -> Path:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(x="Algorithm", y=self.value_col, data=self.df_long,
                    palette="plasma", hue="Algorithm", legend=False, ax=ax)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(self.labelForY())
        ax.tick_params(axis="x", rotation=45, labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        fig.tight_layout()
        return self.save(fig, output_path)

    def violinplot(self, output_path: str = "violinplot.png") -> Path:
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x="Algorithm", y=self.value_col, data=self.df_long,
                       inner="quartile", palette="plasma", hue="Algorithm",
                       legend=False, ax=ax)
        ax.set_title(f"Algorithm {self.value_col} - Violin Plot")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(self.labelForY())
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return self.save(fig, output_path)

    def strip_plot(self, output_path: str = "stripplot.png") -> Path:
        sns.set_theme(style="white")
        fig, ax = plt.subplots(figsize=(14, 14))
        sns.stripplot(x="Algorithm", y=self.value_col, data=self.df_long,
                      jitter=True, palette="colorblind", hue="Algorithm",
                      legend=False, ax=ax)
        ax.set_xlabel("Algorithm", fontsize=25)
        ax.set_ylabel(self.labelForY(), fontsize=25)
        ax.tick_params(axis="both", labelsize=25)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return self.save(fig, output_path)

    def point_plot(self, output_path: str = "pointplot.png") -> Path:
        sns.set_theme(style="dark")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.pointplot(x="Algorithm", y=self.value_col, data=self.df_long,
                      palette="bright", capsize=0.1, hue="Algorithm",
                      legend=False, ax=ax)
        ax.set_title(f"Algorithm {self.value_col} - Point Plot")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(self.labelForY())
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return self.save(fig, output_path)

    def lineplot(self, output_path: str = "lineplot.png",
                 x_col: str = "Image") -> Path:
        sns.set_theme(style="whitegrid")
        # Note: figsize was (100, 60) historically -- almost certainly a typo.
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=x_col, y=self.value_col, hue="Algorithm",
                     marker=".", data=self.df_long, ax=ax)
        ax.set_title(f"Algorithm {self.value_col} per {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(self.labelForY())
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return self.save(fig, output_path)

    def histogram(self, output_path: str = "histogram.png") -> Path:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=self.df_long, x=self.value_col, hue="Algorithm",
                     element="step", stat="density", common_norm=False,
                     palette="pastel", bins=20, ax=ax)
        ax.set_title(f"Distribution of Algorithm {self.value_col}")
        ax.set_xlabel(self.labelForY())
        ax.set_ylabel("Density")
        fig.tight_layout()
        return self.save(fig, output_path)

    # -------- helpers -------------------------------------------------------

    def labelForY(self) -> str:
        col = self.value_col.lower()
        if col == "error":
            return "Euclidean error (pixels)"
        if col == "time":
            return f"{self.value_col} (seconds)"
        if col in ("x", "y"):
            return f"{self.value_col} coordinate (pixels)"
        return self.value_col

    @staticmethod
    def save(fig, output_path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Wrote %s", path)
        return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Input CSV (wide format)")
    parser.add_argument("--out-dir", default=".",
                        help="Where to write the plot PNGs")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose progress logging (INFO level)")
    return parser.parse_args(argv)


def generatePlots(tg: TimesGraph, outDir: Path, tag: str) -> None:
    tg.boxplot(str(outDir / f"{tag}-boxplot.png"))
    tg.violinplot(str(outDir / f"{tag}-violinplot.png"))
    tg.strip_plot(str(outDir / f"{tag}-stripplot.png"))
    tg.point_plot(str(outDir / f"{tag}-pointplot.png"))
    tg.lineplot(str(outDir / f"{tag}-lineplot.png"))
    tg.histogram(str(outDir / f"{tag}-histogram.png"))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    outDir = Path(args.out_dir)

    timesGraph = TimesGraph(args.csv, value_col="Time")
    logger.info("Generating time plots...")
    generatePlots(timesGraph, outDir, "time")

    coordXGraph = TimesGraph(args.csv, value_col="X")
    logger.info("Generating X-coordinate plots...")
    generatePlots(coordXGraph, outDir, "coord-x")

    coordYGraph = TimesGraph(args.csv, value_col="Y")
    logger.info("Generating Y-coordinate plots...")
    generatePlots(coordYGraph, outDir, "coord-y")


if __name__ == "__main__":
    main()
