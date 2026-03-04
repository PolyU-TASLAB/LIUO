#!/usr/bin/env python3
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def default_logs_dir() -> Path:
    return (Path(__file__).resolve().parent.parent / "logs").resolve()


def legacy_log_dir() -> Path:
    return (Path(__file__).resolve().parent.parent / "log").resolve()


def find_latest_timing_csv(log_dirs: List[Path]) -> Path:
    all_candidates = []
    searched_dirs = []

    for log_dir in log_dirs:
        searched_dirs.append(str(log_dir))
        if not log_dir.exists():
            continue
        all_candidates.extend([path for path in log_dir.glob("*timing*.csv") if path.is_file()])

    if not all_candidates:
        raise FileNotFoundError(
            "No timing CSV found. Searched directories: "
            + ", ".join(searched_dirs)
            + ". Expected files like runbag_timing_*.csv"
        )

    return max(all_candidates, key=lambda path: path.stat().st_mtime)


def parse_timestamp_from_filename(csv_path: Path) -> str:
    match = re.search(r"(\d{8}_\d{6})", csv_path.stem)
    if not match:
        return "N/A"
    try:
        dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "N/A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze LIUO RunBag timing CSV and generate plots/report."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to timing CSV. If omitted, auto-select newest *timing*.csv in logs directory.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. If omitted, defaults to logs directory.",
    )
    parser.add_argument(
        "--logs",
        default=str(default_logs_dir()),
        help="Primary logs directory used for auto-discovery and default output.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "frame_idx",
        "status",
        "bag_spin_ms",
        "prepare_msgs_ms",
        "update_one_seg_ms",
        "solve_total_ms",
        "process_liuo_ms",
        "prior_ms",
        "attr_ms",
        "frame_total_ms",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def create_plots(df_ok: pd.DataFrame, out_dir: Path) -> dict:
    artifacts = {}

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_ok["frame_idx"], df_ok["frame_total_ms"], linewidth=1.2)
    ax.set_title("RunBag Frame Total Latency")
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("ms")
    ax.grid(True, linestyle="--", alpha=0.3)
    path = out_dir / "frame_total_trend.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    artifacts["trend"] = path

    stage_columns = [
        "bag_spin_ms",
        "prepare_msgs_ms",
        "update_one_seg_ms",
        "solve_total_ms",
    ]
    stage_means = df_ok[stage_columns].mean().fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.pie(
        stage_means.values,
        labels=stage_columns,
        autopct="%1.1f%%",
        startangle=120,
    )
    ax.set_title("Average Stage Time Share")
    path = out_dir / "stage_share_pie.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    artifacts["share"] = path

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot([df_ok[col].dropna().values for col in stage_columns], labels=stage_columns)
    ax.set_title("Stage Latency Distribution")
    ax.set_ylabel("ms")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    path = out_dir / "stage_latency_boxplot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    artifacts["boxplot"] = path

    return artifacts


def build_recommendations(df_ok: pd.DataFrame) -> list:
    recommendations = []

    p50 = float(df_ok["frame_total_ms"].quantile(0.5))
    p99 = float(df_ok["frame_total_ms"].quantile(0.99))
    if p50 > 0 and p99 > 2.0 * p50:
        recommendations.append(
            "P99 明显高于 P50，存在抖动；建议先排查慢帧对应时刻的 bag I/O 与点云关联阶段。"
        )

    stage_columns = [
        "bag_spin_ms",
        "prepare_msgs_ms",
        "update_one_seg_ms",
        "solve_total_ms",
    ]
    stage_means = df_ok[stage_columns].mean().fillna(0.0)
    dominant = stage_means.idxmax()
    recommendations.append(
        f"主耗时阶段为 {dominant}；优先优化该阶段可获得最直接收益。"
    )

    if dominant == "solve_total_ms":
        recommendations.append(
            "优化求解器相关参数（如迭代次数、残差数量、关键帧局部地图大小）以降低优化耗时。"
        )
    elif dominant == "prepare_msgs_ms":
        recommendations.append(
            "数据组帧阶段偏重，建议检查消息过滤、时间对齐与缓存清理逻辑，避免不必要遍历。"
        )
    elif dominant == "bag_spin_ms":
        recommendations.append(
            "bag 读取开销较高，可考虑 SSD、减小无关话题读取、或提前离线拆包。"
        )
    else:
        recommendations.append(
            "控制点更新阶段较重，建议检查 knot 密度策略，避免在平稳段过密插点。"
        )

    return recommendations[:3]


def write_report(df: pd.DataFrame, df_ok: pd.DataFrame, out_dir: Path, artifacts: dict) -> Path:
    report_path = out_dir / "timing_report.md"

    total_frames = len(df)
    ok_frames = len(df_ok)
    dropped = total_frames - ok_frames

    p50 = float(df_ok["frame_total_ms"].quantile(0.5)) if ok_frames else float("nan")
    p90 = float(df_ok["frame_total_ms"].quantile(0.9)) if ok_frames else float("nan")
    p99 = float(df_ok["frame_total_ms"].quantile(0.99)) if ok_frames else float("nan")
    mean = float(df_ok["frame_total_ms"].mean()) if ok_frames else float("nan")

    slow_frames = df_ok.nlargest(10, "frame_total_ms")[[
        "frame_idx",
        "frame_total_ms",
        "bag_spin_ms",
        "prepare_msgs_ms",
        "update_one_seg_ms",
        "solve_total_ms",
    ]]

    recommendations = build_recommendations(df_ok) if ok_frames else ["没有有效帧可分析。"]

    lines = [
        "# RunBag Timing Report",
        "",
        "## Summary",
        f"- Total rows: {total_frames}",
        f"- Valid rows (`status == ok`): {ok_frames}",
        f"- Non-ok rows: {dropped}",
        f"- Frame total latency mean: {mean:.3f} ms" if ok_frames else "- Frame total latency mean: N/A",
        f"- P50: {p50:.3f} ms" if ok_frames else "- P50: N/A",
        f"- P90: {p90:.3f} ms" if ok_frames else "- P90: N/A",
        f"- P99: {p99:.3f} ms" if ok_frames else "- P99: N/A",
        "",
        "## Artifacts",
        f"- Trend: {artifacts['trend'].name}",
        f"- Stage share: {artifacts['share'].name}",
        f"- Stage boxplot: {artifacts['boxplot'].name}",
        "",
        "## Top Slow Frames",
        "| frame_idx | frame_total_ms | bag_spin_ms | prepare_msgs_ms | update_one_seg_ms | solve_total_ms |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    if ok_frames:
        for _, row in slow_frames.iterrows():
            lines.append(
                f"| {int(row['frame_idx'])} | {row['frame_total_ms']:.3f} | {row['bag_spin_ms']:.3f} | {row['prepare_msgs_ms']:.3f} | {row['update_one_seg_ms']:.3f} | {row['solve_total_ms']:.3f} |"
            )
    else:
        lines.append("| - | - | - | - | - | - |")

    lines.extend(["", "## Recommendations"])
    for suggestion in recommendations:
        lines.append(f"- {suggestion}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    logs_dir = Path(args.logs).expanduser().resolve()
    candidate_log_dirs = [logs_dir]
    if logs_dir != legacy_log_dir():
        candidate_log_dirs.append(legacy_log_dir())

    if args.input:
        input_path = Path(args.input).expanduser().resolve()
    else:
        input_path = find_latest_timing_csv(candidate_log_dirs)

    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    else:
        out_dir = logs_dir if logs_dir.exists() else input_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input CSV: {input_path}")
    print(f"[INFO] Input CSV timestamp(from name): {parse_timestamp_from_filename(input_path)}")
    print(
        "[INFO] Input CSV mtime: "
        + datetime.fromtimestamp(input_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    )
    print(f"[INFO] Output dir: {out_dir}")

    df = pd.read_csv(input_path)
    df = ensure_columns(df)
    df_ok = df[df["status"] == "ok"].copy()

    numeric_columns = [
        "bag_spin_ms",
        "prepare_msgs_ms",
        "update_one_seg_ms",
        "solve_total_ms",
        "process_liuo_ms",
        "prior_ms",
        "attr_ms",
        "frame_total_ms",
    ]
    for column in numeric_columns:
        df_ok[column] = pd.to_numeric(df_ok[column], errors="coerce")

    if df_ok.empty:
        raise RuntimeError("No valid rows (status == ok) found in input CSV.")

    artifacts = create_plots(df_ok, out_dir)
    report_path = write_report(df, df_ok, out_dir, artifacts)

    print(f"[OK] Report: {report_path}")
    for key, value in artifacts.items():
        print(f"[OK] {key}: {value}")


if __name__ == "__main__":
    main()
