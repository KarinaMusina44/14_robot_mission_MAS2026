from __future__ import annotations
from model import RobotMissionModel
from mesa.batchrunner import batch_run
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import os
from pathlib import Path
from typing import List
import math

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_green_waste_values(raw: str) -> List[int]:
    values = _parse_int_list(raw)
    invalid = [v for v in values if v % 4 != 0]
    if invalid:
        raise ValueError(
            f"Invalid --n-waste values (must be divisible by 4): {invalid}"
        )
    return values


def _pick_reference_value(values: list[int]) -> int:
    unique_sorted = sorted(set(values))
    return unique_sorted[len(unique_sorted) // 2]


def _filter_by_fixed_values(
    completion_df: pd.DataFrame, fixed_values: dict[str, int]
) -> pd.DataFrame:
    filtered = completion_df
    for col, value in fixed_values.items():
        filtered = filtered.loc[filtered[col] == value]
    return filtered


def _fixed_values_label(fixed_values: dict[str, int]) -> str:
    display_names = {
        "n_green_robots": "green",
        "n_red_robots": "red",
        "n_yellow_robots": "yellow",
        "n_waste": "waste",
    }
    parts = [f"{display_names.get(col, col)}={value}" for col, value in fixed_values.items()]
    return ", ".join(parts)


def _aggregate_time_by_agent_count(
    completion_df: pd.DataFrame, count_col: str, fixed_values: dict[str, int]
) -> pd.DataFrame:
    filtered = _filter_by_fixed_values(completion_df, fixed_values=fixed_values)
    completed = filtered.loc[
        filtered["completed"], [count_col, "time_to_clear_all_waste"]
    ].copy()
    if completed.empty:
        return pd.DataFrame(
            columns=[count_col, "mean_time_to_clear",
                     "std_time_to_clear", "completed_runs"]
        )

    aggregated = (
        completed.groupby(count_col, dropna=False)["time_to_clear_all_waste"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_time_to_clear",
                "std": "std_time_to_clear",
                "count": "completed_runs",
            }
        )
        .sort_values(count_col)
        .reset_index(drop=True)
    )
    return aggregated


def _save_time_vs_agent_count_plot(
    completion_df: pd.DataFrame,
    count_col: str,
    x_label: str,
    fixed_values: dict[str, int],
    line_color: str,
    fill_color: str,
    plot_path: Path,
) -> None:
    agg = _aggregate_time_by_agent_count(
        completion_df, count_col=count_col, fixed_values=fixed_values
    )
    if agg.empty:
        return

    x = agg[count_col].astype(float).tolist()
    y = agg["mean_time_to_clear"].astype(float).tolist()
    std = agg["std_time_to_clear"].fillna(0.0).astype(float).tolist()
    n = agg["completed_runs"].astype(int).tolist()

    half_width = []
    for s, count in zip(std, n):
        if count <= 1:
            half_width.append(0.0)
        else:
            half_width.append(1.96 * (s / math.sqrt(count)))
    band_label = "95% CI"

    lower = [mean - width for mean, width in zip(y, half_width)]
    upper = [mean + width for mean, width in zip(y, half_width)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color=line_color, linewidth=2.0, marker="o", label="Mean")
    ax.fill_between(x, lower, upper, color=fill_color,
                    alpha=0.22, label=band_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("mean_time_to_clear_all_waste")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Time-to-Clear vs {x_label}\nfixed: {_fixed_values_label(fixed_values)}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch experiments and compute time to clear all wastes."
    )
    parser.add_argument("--width", type=int, default=20, help="Grid width.")
    parser.add_argument("--height", type=int, default=10, help="Grid height.")
    parser.add_argument(
        "--n-waste",
        type=str,
        default="32",
        help="Comma-separated values for initial green waste count (must be divisible by 4).",
    )
    parser.add_argument(
        "--n-green-robots",
        type=str,
        default="3",
        help="Comma-separated values for number of green robots.",
    )
    parser.add_argument(
        "--n-yellow-robots",
        type=str,
        default="2",
        help="Comma-separated values for number of yellow robots.",
    )
    parser.add_argument(
        "--n-red-robots",
        type=str,
        default="1",
        help="Comma-separated values for number of red robots.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="How many runs per parameter combination.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed used to build the seed list.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Maximum number of model steps per run.",
    )
    parser.add_argument(
        "--data-period",
        type=int,
        default=1,
        help="Data collection period for batch_run (1 = every step).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of processes for batch_run.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="batch_results",
        help="Output directory for CSV result files.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress display.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation.",
    )
    return parser.parse_args()


def _extract_completion_per_run(
    df: pd.DataFrame,
    config_cols: list[str],
    time_col: str = "time",
    remaining_col: str = "system_waste_total",
) -> pd.DataFrame:
    rows = []
    for run_id, run_df in df.groupby("RunId", sort=True):
        run_sorted = run_df.sort_values(time_col)
        done_rows = run_sorted[run_sorted[remaining_col] <= 0]
        completed = not done_rows.empty

        if completed:
            completion_time = float(done_rows.iloc[0][time_col])
        else:
            completion_time = float("nan")

        last_row = run_sorted.iloc[-1]
        row = {
            "RunId": int(run_id),
            "iteration": int(last_row.get("iteration", 0)),
            "seed": last_row.get("seed", pd.NA),
            "completed": bool(completed),
            "time_to_clear_all_waste": completion_time,
            "final_time": float(last_row[time_col]),
            "final_system_waste_total": float(last_row[remaining_col]),
        }
        for col in config_cols:
            row[col] = last_row[col]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    parameters = {
        "width": [args.width],
        "height": [args.height],
        "n_waste": _parse_green_waste_values(args.n_waste),
        "n_green_robots": _parse_int_list(args.n_green_robots),
        "n_yellow_robots": _parse_int_list(args.n_yellow_robots),
        "n_red_robots": _parse_int_list(args.n_red_robots),
    }
    seed_values = [args.seed_start + i for i in range(max(1, args.iterations))]

    print("Running batch experiments...")
    print(f"parameters={parameters}")
    print(f"seeds={seed_values}")
    print(
        f"max_steps={args.max_steps}, data_period={args.data_period}, processes={args.processes}"
    )

    results = batch_run(
        RobotMissionModel,
        parameters=parameters,
        number_processes=args.processes,
        rng=seed_values,
        max_steps=args.max_steps,
        data_collection_period=args.data_period,
        display_progress=not args.no_progress,
    )
    if not results:
        raise RuntimeError("No results produced by batch_run.")

    df = pd.DataFrame(results)
    raw_csv = outdir / "batch_raw.csv"
    df.to_csv(raw_csv, index=False)
    print(f"Saved raw results: {raw_csv}")

    time_col = "time"
    remaining_col = "system_waste_total"
    if time_col not in df.columns:
        raise ValueError(
            f"Column '{time_col}' not found. Available columns: {sorted(df.columns)}"
        )
    if remaining_col not in df.columns:
        raise ValueError(
            f"Column '{remaining_col}' not found. Available columns: {sorted(df.columns)}"
        )

    config_cols = [
        "width",
        "height",
        "n_waste",
        "n_green_robots",
        "n_yellow_robots",
        "n_red_robots",
    ]
    completion_df = _extract_completion_per_run(
        df=df,
        config_cols=config_cols,
        time_col=time_col,
        remaining_col=remaining_col,
    )
    completion_csv = outdir / "batch_completion_per_run.csv"
    completion_df.to_csv(completion_csv, index=False)
    print(f"Saved per-run completion times: {completion_csv}")

    completed_total = int(completion_df["completed"].sum())
    runs_total = int(len(completion_df))
    completion_rate = float(
        completed_total / runs_total) if runs_total > 0 else 0.0
    completed_times = completion_df.loc[completion_df["completed"],
                                        "time_to_clear_all_waste"]
    overall_mean = float(completed_times.mean()) if len(
        completed_times) > 0 else float("nan")
    overall_std = float(completed_times.std(ddof=1)) if len(
        completed_times) > 1 else float("nan")
    print(
        "Overall: "
        f"runs={runs_total}, completed={completed_total}, completion_rate={completion_rate:.3f}, "
        f"mean_time_to_clear={overall_mean}, std_time_to_clear={overall_std}"
    )

    if args.no_plots:
        print("Plot generation disabled (--no-plots).")
        return 0

    reference_values = {
        "n_green_robots": _pick_reference_value(parameters["n_green_robots"]),
        "n_red_robots": _pick_reference_value(parameters["n_red_robots"]),
        "n_yellow_robots": _pick_reference_value(parameters["n_yellow_robots"]),
        "n_waste": _pick_reference_value(parameters["n_waste"]),
    }
    print(f"Fixed reference values (median): {reference_values}")

    per_color_specs = [
        (
            "n_green_robots",
            "Number of green robots",
            "#2A9D8F",
            "#76C893",
            outdir / "plot_time_to_clear_vs_green_agents.png",
        ),
        (
            "n_red_robots",
            "Number of red robots",
            "#C1121F",
            "#E63946",
            outdir / "plot_time_to_clear_vs_red_agents.png",
        ),
        (
            "n_yellow_robots",
            "Number of yellow robots",
            "#EE9B00",
            "#FFD166",
            outdir / "plot_time_to_clear_vs_yellow_agents.png",
        ),
        (
            "n_waste",
            "Number of initial waste units",
            "#264653",
            "#2A9D8F",
            outdir / "plot_time_to_clear_vs_waste.png",
        ),
    ]

    for count_col, x_label, line_color, fill_color, plot_path in per_color_specs:
        fixed_values = {
            col: value for col, value in reference_values.items() if col != count_col
        }
        _save_time_vs_agent_count_plot(
            completion_df=completion_df,
            count_col=count_col,
            x_label=x_label,
            fixed_values=fixed_values,
            line_color=line_color,
            fill_color=fill_color,
            plot_path=plot_path,
        )
        if plot_path.exists():
            print(f"Saved plot: {plot_path}")
        else:
            print(f"No completed runs: skipped {count_col} plot.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
