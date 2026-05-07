#!/usr/bin/env python3
"""Plot GPU resource-consumption figures from monitor_gpu_timeseries CSV files."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "elapsed_sec",
    "gpu_index",
    "gpu_util_pct",
    "gpu_memory_used_mib",
}

COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
}


def parse_float(value: str | None) -> float:
    text = str(value or "").strip()
    if text in {"", "[N/A]", "N/A"}:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def parse_gpu_list(value: str) -> list[int]:
    gpus: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            gpu = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid GPU index: {part}") from exc
        if gpu not in gpus:
            gpus.append(gpu)
    if not gpus:
        raise argparse.ArgumentTypeError("--gpus must include at least one GPU index")
    return gpus


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def default_output_prefix(csv_path: Path) -> Path:
    stem = csv_path.stem
    suffix = stem.removeprefix("gpu_trace_")
    return csv_path.with_name(f"gpu_resource_consumption_{suffix}")


def load_gpu_series(csv_path: Path, selected_gpus: Iterable[int]) -> dict[int, dict[str, list[float]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    selected = list(selected_gpus)
    selected_set = set(selected)
    series = {
        gpu: {
            "elapsed_sec": [],
            "gpu_util_pct": [],
            "gpu_memory_used_mib": [],
        }
        for gpu in selected
    }

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        for row in reader:
            gpu = int(parse_float(row.get("gpu_index")))
            if gpu not in selected_set:
                continue
            series[gpu]["elapsed_sec"].append(parse_float(row.get("elapsed_sec")))
            series[gpu]["gpu_util_pct"].append(parse_float(row.get("gpu_util_pct")))
            series[gpu]["gpu_memory_used_mib"].append(parse_float(row.get("gpu_memory_used_mib")))

    nonempty: dict[int, dict[str, list[float]]] = {}
    for gpu in selected:
        values = series[gpu]
        if not values["elapsed_sec"]:
            print(f"warning: GPU {gpu} has no rows in {csv_path}; skipping", file=sys.stderr)
            continue

        order = sorted(range(len(values["elapsed_sec"])), key=lambda idx: values["elapsed_sec"][idx])
        nonempty[gpu] = {
            key: [values[key][idx] for idx in order]
            for key in values
        }

    if not nonempty:
        raise ValueError(f"none of the requested GPUs were found: {', '.join(map(str, selected))}")

    return nonempty


def compute_time_ranges(
    series: dict[int, dict[str, list[float]]],
    use_broken_axis: bool,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    times = [value for gpu_data in series.values() for value in gpu_data["elapsed_sec"]]
    min_time = min(times)
    max_time = max(times)
    duration = max_time - min_time
    if not use_broken_axis or duration <= 0:
        return None

    left_end = min_time + duration * 0.28
    right_start = max_time - duration * 0.22
    if left_end >= right_start:
        return None

    pad = max(duration * 0.01, 1.0)
    return (min_time - pad, left_end), (right_start, max_time + pad)


def color_for_gpu(gpu: int, position: int) -> str:
    if gpu in COLORS:
        return COLORS[gpu]
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return fallback[position % len(fallback)]


def plot_metric(
    axes: list[plt.Axes],
    series: dict[int, dict[str, list[float]]],
    metric: str,
    ylabel: str,
    ylimit: tuple[float, float],
) -> None:
    for position, (gpu, values) in enumerate(series.items()):
        color = color_for_gpu(gpu, position)
        for ax in axes:
            ax.plot(
                values["elapsed_sec"],
                values[metric],
                label=f"GPU {gpu}",
                color=color,
                linewidth=0.9,
                alpha=0.95,
            )

    for ax in axes:
        ax.set_ylim(*ylimit)
        ax.grid(False)
        ax.tick_params(direction="out", length=3, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    axes[0].set_ylabel(ylabel)


def add_break_marks(left_ax: plt.Axes, right_ax: plt.Axes) -> None:
    d = 0.018
    top_kwargs = {
        "color": "#C62828",
        "clip_on": False,
        "linewidth": 1.0,
        "transform": left_ax.transAxes,
    }
    bottom_kwargs = {
        "color": "#2E7D32",
        "clip_on": False,
        "linewidth": 1.0,
        "transform": left_ax.transAxes,
    }
    left_ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **top_kwargs)
    left_ax.plot((1 - d, 1 + d), (-d, d), **bottom_kwargs)

    top_kwargs["transform"] = right_ax.transAxes
    bottom_kwargs["transform"] = right_ax.transAxes
    right_ax.plot((-d, d), (1 - d, 1 + d), **top_kwargs)
    right_ax.plot((-d, d), (-d, d), **bottom_kwargs)


def configure_broken_pair(
    left_ax: plt.Axes,
    right_ax: plt.Axes,
    ranges: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    left_range, right_range = ranges
    left_ax.set_xlim(*left_range)
    right_ax.set_xlim(*right_range)
    left_ax.spines["right"].set_visible(False)
    right_ax.spines["left"].set_visible(False)
    right_ax.tick_params(axis="y", left=False, labelleft=False)
    add_break_marks(left_ax, right_ax)


def wrap_caption(caption: str) -> str:
    return "\n".join(textwrap.wrap(caption, width=86, break_long_words=False))


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, bbox_inches="tight")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        fig.savefig(fallback, bbox_inches="tight")
        print(f"warning: could not overwrite locked file: {path}", file=sys.stderr)
        return fallback


def build_figure(
    series: dict[int, dict[str, list[float]]],
    broken_axis: bool,
    caption: str | None,
) -> plt.Figure:
    all_util = [value for values in series.values() for value in values["gpu_util_pct"]]
    all_memory = [value for values in series.values() for value in values["gpu_memory_used_mib"]]
    util_top = max(100.0, max(all_util) * 1.08 if all_util else 100.0)
    memory_top = max(1.0, max(all_memory) * 1.12 if all_memory else 1.0)
    time_ranges = compute_time_ranges(series, broken_axis)

    if time_ranges:
        fig = plt.figure(figsize=(6.4, 5.65))
        grid = fig.add_gridspec(
            2,
            2,
            left=0.11,
            right=0.97,
            top=0.95,
            bottom=0.18 if caption else 0.14,
            hspace=0.72,
            wspace=0.05,
            width_ratios=(1.0, 1.0),
        )
        util_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
        memory_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

        plot_metric(util_axes, series, "gpu_util_pct", "GPU Usage (%)", (0, util_top))
        plot_metric(memory_axes, series, "gpu_memory_used_mib", "Memory Usage (MiB)", (0, memory_top))
        configure_broken_pair(util_axes[0], util_axes[1], time_ranges)
        configure_broken_pair(memory_axes[0], memory_axes[1], time_ranges)
        util_axes[1].legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#222222")
        memory_axes[1].legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#222222")
    else:
        fig, (util_ax, memory_ax) = plt.subplots(2, 1, figsize=(6.4, 5.65), sharex=True)
        fig.subplots_adjust(
            left=0.11,
            right=0.97,
            top=0.95,
            bottom=0.18 if caption else 0.14,
            hspace=0.72,
        )
        util_axes = [util_ax]
        memory_axes = [memory_ax]
        plot_metric(util_axes, series, "gpu_util_pct", "GPU Usage (%)", (0, util_top))
        plot_metric(memory_axes, series, "gpu_memory_used_mib", "Memory Usage (MiB)", (0, memory_top))
        util_ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#222222")
        memory_ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#222222")

    fig.text(0.5, 0.535, "(a) GPU Utilization (%) Over Time (s)", ha="center", va="center", fontsize=12)
    fig.text(0.5, 0.105 if caption else 0.07, "(b) GPU Memory Usage (MiB) Over Time (s)", ha="center", va="center", fontsize=12)
    if caption:
        fig.text(0.5, 0.035, wrap_caption(caption), ha="center", va="center", fontsize=11)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a paper-style GPU resource-consumption figure from gpu_trace CSV data."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Input gpu_trace_*.csv path")
    parser.add_argument(
        "--gpus",
        type=parse_gpu_list,
        default=parse_gpu_list("0,2,5,6"),
        help="Comma-separated GPU indices to plot, default: 0,2,5,6",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output prefix without extension; defaults to gpu_resource_consumption_<timestamp> next to the CSV",
    )
    parser.add_argument("--broken-axis", action="store_true", help="Use a two-segment broken x-axis")
    parser.add_argument("--caption", help="Optional figure caption rendered below the two panels")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.csv.expanduser()
    output_prefix = args.output_prefix.expanduser() if args.output_prefix else default_output_prefix(csv_path)

    setup_style()
    series = load_gpu_series(csv_path, args.gpus)
    fig = build_figure(series, args.broken_axis, args.caption)
    png_path = save_figure(fig, output_prefix.with_suffix(".png"))
    pdf_path = save_figure(fig, output_prefix.with_suffix(".pdf"))
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
