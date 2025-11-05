from __future__ import annotations
import argparse, csv, os, re, sys
from itertools import combinations
from typing import Iterable, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


def parse_hhmmss_msec(s: str) -> float:
    s = s.strip().replace(",", ".")
    m = re.fullmatch(r"(?:(\d+):)?([0-5]?\d):([0-5]?\d)(?:\.(\d{1,3}))?", s)
    if not m:
        raise ValueError(f"Invalid time string: {s}")
    h = int(m.group(1)) if m.group(1) else 0
    mi = int(m.group(2)); sec = int(m.group(3))
    ms = int((m.group(4) or "0").ljust(3, "0"))
    return h*3600 + mi*60 + sec + ms/1000.0


def fmt_hhmmss_comma_ms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    s_int = int(s)
    ms = int(round((s - s_int) * 1000))
    if ms == 1000:
        ms = 0; s_int += 1
        if s_int == 60:
            s_int = 0; m += 1
            if m == 60:
                m = 0; h += 1
    return f"{h:02d}:{m:02d}:{s_int:02d},{ms:03d}"


def discover_ids(columns: Iterable[str]) -> List[int]:
    ids = set()
    for c in columns:
        m = re.fullmatch(r'[xy](\d+)', c)
        if m: ids.add(int(m.group(1)))
    return sorted(ids)


def segment_mask_frames(mask, min_len_frames: int):
    segs = []
    in_seg = False; s = 0
    for i, m in enumerate(mask):
        if m and not in_seg:
            in_seg = True; s = i
        elif not m and in_seg:
            e = i - 1
            if (e - s + 1) >= min_len_frames:
                segs.append((s, e))
            in_seg = False
    if in_seg:
        e = len(mask) - 1
        if (e - s + 1) >= min_len_frames:
            segs.append((s, e))
    return segs


def time_from_frame(frame_index: int, fps: float, anchor_offset_s: float = 0.0) -> float:
    return frame_index / fps + anchor_offset_s


def compute_anchor_offset(anchor_frame, anchor_time_str, fps: float) -> float:
    if anchor_frame is None or anchor_time_str is None:
        return 0.0
    desired_s = parse_hhmmss_msec(anchor_time_str)
    raw_s = anchor_frame / fps
    return desired_s - raw_s


def process(csv_path: str, output_prefix: str | None, fps: float, threshold_px: float,
            min_frames: int, export_distances: bool, export_frames_only: bool,
            export_inqscribe: bool, include_nan_events: bool,
            export_per_id_distance: bool, export_trajectory_plot: bool,
            anchor_frame: int | None, anchor_time: str | None) -> Dict[str, str]:

    df = pd.read_csv(csv_path)
    ids = discover_ids(df.columns)
    if not ids:
        raise ValueError("No xN/yN columns found (e.g., x1,y1).")

    n = len(df)
    frames = np.arange(n, dtype=int)
    anchor_offset_s = compute_anchor_offset(anchor_frame, anchor_time, fps)

    valid_by_id = {i: ~(df.get(f"x{i}").isna() | df.get(f"y{i}").isna()) for i in ids}

    events = []
    frame_events_rows = []
    distance_rows = []
    per_id_rows: List[Dict[str, float]] = []

    for i, j in combinations(ids, 2):
        xi, yi = df[f"x{i}"].astype(float).values, df[f"y{i}"].astype(float).values
        xj, yj = df[f"x{j}"].astype(float).values, df[f"y{j}"].astype(float).values
        valid_both = (valid_by_id[i].values) & (valid_by_id[j].values)
        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)

        prox_mask = (dist <= threshold_px) & valid_both
        prox_segs = segment_mask_frames(prox_mask, min_frames)

        for s, e in prox_segs:
            start_s = time_from_frame(s, fps, anchor_offset_s)
            end_s = time_from_frame(e, fps, anchor_offset_s)
            title = f"Proximity: ID{i}-ID{j} [F{s}–F{e}]"
            comment = f"distance ≤ {int(threshold_px)}px; duration={e - s + 1} frames; both valid"
            events.append((start_s, end_s, title, comment))
            if export_frames_only:
                frame_events_rows.append([s, e, e - s + 1, f"proximity_{i}_{j}", "distance<=threshold & both valid"])

        if export_distances:
            for f in range(n):
                t_sec = time_from_frame(f, fps, anchor_offset_s)
                distance_rows.append([f, i, j, t_sec, dist[f], int(valid_both[f]), int((dist[f] <= threshold_px) and valid_both[f])])

    if include_nan_events:
        # "both-NaN" across all IDs
        both_nan = np.ones(n, dtype=bool)
        for i in ids:
            both_nan &= ~valid_by_id[i].values
        # per-ID only (non-overlapping with both)
        id_only = {i: ((~valid_by_id[i].values) & (~both_nan)) for i in ids}

        for s, e in segment_mask_frames(both_nan, min_frames):
            start_s = time_from_frame(s, fps, anchor_offset_s)
            end_s = time_from_frame(e, fps, anchor_offset_s)
            events.append((start_s, end_s, f"NaN: ALL IDs missing [F{s}–F{e}]", f"duration={e - s + 1} frames"))
            if export_frames_only:
                frame_events_rows.append([s, e, e - s + 1, "nan_all", "all IDs missing"])

        for i in ids:
            for s, e in segment_mask_frames(id_only[i], min_frames):
                start_s = time_from_frame(s, fps, anchor_offset_s)
                end_s = time_from_frame(e, fps, anchor_offset_s)
                events.append((start_s, end_s, f"NaN: ID{i} missing [F{s}–F{e}]", f"duration={e - s + 1} frames"))
                if export_frames_only:
                    frame_events_rows.append([s, e, e - s + 1, f"nan_id{i}", f"ID{i} missing"])

    events.sort(key=lambda r: (r[0], r[1], r[2]))

    base = output_prefix or os.path.splitext(csv_path)[0]
    artifacts: Dict[str, str] = {}

    if export_inqscribe:
        inq_path = f"{base}_InqScribe_thresh{int(threshold_px)}px_{fps:.3f}fps.txt"
        with open(inq_path, "w", encoding="utf-8-sig", newline="\n") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
            w.writerow(["Start Time", "End Time", "Title", "Comment"])
            for start_s, end_s, title, comment in events:
                w.writerow([fmt_hhmmss_comma_ms(start_s), fmt_hhmmss_comma_ms(end_s), title, comment])
        artifacts["inqscribe"] = inq_path

    if export_frames_only:
        frames_path = f"{base}_EventsByFrame_thresh{int(threshold_px)}px.tsv"
        fr_df = pd.DataFrame(frame_events_rows, columns=["start_frame","end_frame","len_frames","type","note"])
        fr_df.sort_values(["start_frame","end_frame","type"], inplace=True)
        fr_df.to_csv(frames_path, sep="\t", index=False, encoding="utf-8")
        artifacts["frames_only"] = frames_path

    if export_distances:
        dist_path = f"{base}_PairwiseDistances_{int(threshold_px)}px_{fps:.3f}fps.csv"
        dist_df = pd.DataFrame(distance_rows, columns=["frame","id_i","id_j","t_s","distance_px","both_valid",f"contact_le_{int(threshold_px)}px"])
        dist_df.to_csv(dist_path, index=False, encoding="utf-8")
        artifacts["distances"] = dist_path

    if export_per_id_distance:
        for i in ids:
            xi = df[f"x{i}"].astype(float).to_numpy()
            yi = df[f"y{i}"].astype(float).to_numpy()
            valid = valid_by_id[i].values.astype(bool)
            valid_frames = int(valid.sum())

            dx = np.diff(xi)
            dy = np.diff(yi)
            step_mask = valid[:-1] & valid[1:]
            step_dist = np.sqrt(dx**2 + dy**2)
            total_distance_px = float(step_dist[step_mask].sum())
            steps_used = int(step_mask.sum())

            per_id_rows.append({
                "id": i,
                "frames_with_valid_xy": valid_frames,
                "steps_used": steps_used,
                "path_length_px": total_distance_px,
                "mean_step_px": float(total_distance_px / steps_used) if steps_used else 0.0,
            })

        per_id_df = pd.DataFrame(per_id_rows, columns=[
            "id",
            "frames_with_valid_xy",
            "steps_used",
            "path_length_px",
            "mean_step_px",
        ])
        per_id_df.sort_values("id", inplace=True)

        per_id_path = f"{base}_PerIDPathLength.csv"
        per_id_df.to_csv(per_id_path, index=False, encoding="utf-8")
        artifacts["per_id_path_length"] = per_id_path

    if export_trajectory_plot and ids:
        traj_path = f"{base}_Trajectories.pdf"
        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = plt.cm.get_cmap("tab20", len(ids))
        any_plotted = False

        for idx, i in enumerate(ids):
            xi = df[f"x{i}"].astype(float).to_numpy()
            yi = df[f"y{i}"].astype(float).to_numpy()
            valid = valid_by_id[i].values.astype(bool)

            if not valid.any():
                continue

            x_plot = np.where(valid, xi, np.nan)
            y_plot = np.where(valid, yi, np.nan)
            color = cmap(idx)
            ax.plot(x_plot, y_plot, label=f"ID{i}", linewidth=1.2, color=color)

            first_idx = np.argmax(valid)
            last_idx = len(valid) - 1 - np.argmax(valid[::-1])
            ax.scatter(xi[first_idx], yi[first_idx], color=color, marker="o", s=25, zorder=3)
            ax.scatter(xi[last_idx], yi[last_idx], color=color, marker="x", s=35, zorder=3)
            any_plotted = True

        ax.set_title("Trajectories by ID")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        if any_plotted:
            ax.legend(loc="upper right", frameon=True, fontsize="small")
        fig.tight_layout()
        fig.savefig(traj_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        artifacts["trajectory_plot"] = traj_path

    return artifacts


def launch_gui():
    """Launch a lightweight Tkinter GUI to gather inputs before running process()."""

    def choose_file():
        file_path = filedialog.askopenfilename(
            title="Select trajectories.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            csv_var.set(file_path)

    def run_process():
        csv_path = csv_var.get().strip()
        if not csv_path:
            messagebox.showerror("Missing File", "Please choose a trajectories CSV file.")
            return

        try:
            threshold_val = float(threshold_var.get())
        except ValueError:
            messagebox.showerror("Invalid Threshold", "Threshold must be a number.")
            return

        try:
            min_frames_val = int(min_frames_var.get())
        except ValueError:
            messagebox.showerror("Invalid Frames", "Frames near must be an integer.")
            return

        if min_frames_val <= 0:
            messagebox.showerror("Invalid Frames", "Frames near must be greater than zero.")
            return

        root.config(cursor="wait")
        root.update_idletasks()
        try:
            artifacts = process(
                csv_path=csv_path,
                output_prefix=None,
                fps=30.0,
                threshold_px=threshold_val,
                min_frames=min_frames_val,
                export_distances=True,
                export_frames_only=True,
                export_inqscribe=True,
                include_nan_events=True,
                export_per_id_distance=True,
                export_trajectory_plot=True,
                anchor_frame=None,
                anchor_time=None,
            )
        except Exception as exc:
            messagebox.showerror("Processing Error", f"Failed to process file:\n{exc}")
        else:
            if artifacts:
                created = "\n".join(f"- {path}" for path in artifacts.values())
            else:
                created = "No outputs were generated."
            messagebox.showinfo("Processing Complete", f"Artifacts created:\n{created}")
        finally:
            root.config(cursor="")
            root.update_idletasks()

    root = tk.Tk()
    root.title("IDTracker InqScribe Helper")

    csv_var = tk.StringVar()
    threshold_var = tk.StringVar(value="60")
    min_frames_var = tk.StringVar(value="6")

    tk.Label(root, text="Trajectories CSV:").grid(row=0, column=0, sticky="w", padx=8, pady=(10, 4))
    entry_csv = tk.Entry(root, textvariable=csv_var, width=40)
    entry_csv.grid(row=0, column=1, padx=(0, 4), pady=(10, 4), sticky="we")
    tk.Button(root, text="Browse…", command=choose_file).grid(row=0, column=2, padx=(0, 10), pady=(10, 4))

    tk.Label(root, text="Distance threshold (px):").grid(row=1, column=0, sticky="w", padx=8, pady=4)
    tk.Entry(root, textvariable=threshold_var, width=10).grid(row=1, column=1, sticky="w", padx=(0, 4), pady=4)

    tk.Label(root, text="Frames near (min length):").grid(row=2, column=0, sticky="w", padx=8, pady=4)
    tk.Entry(root, textvariable=min_frames_var, width=10).grid(row=2, column=1, sticky="w", padx=(0, 4), pady=4)

    tk.Button(root, text="Run", command=run_process).grid(row=3, column=1, pady=(8, 12), sticky="e")
    tk.Button(root, text="Close", command=root.destroy).grid(row=3, column=2, pady=(8, 12), padx=(0, 10))

    root.columnconfigure(1, weight=1)
    root.mainloop()


def main():
    if len(sys.argv) == 1:
        launch_gui()
        return

    p = argparse.ArgumentParser(description="Generate InqScribe + QA outputs from IDTracker trajectories.csv (frame-based, CFR).")
    p.add_argument("csv", help="Path to trajectories.csv (must include xN,yN columns).")
    p.add_argument("--fps", type=float, default=30.0, help="Constant FPS for timecodes (default: 30.0).")
    p.add_argument("--threshold", type=float, default=60.0, help="Proximity threshold in pixels (default: 60).")
    p.add_argument("--min-frames", type=int, default=6, help="Minimum contiguous frames for an event (default: 6).")
    p.add_argument("--output-prefix", type=str, default=None, help="Prefix for output files (default: input base name).")
    p.add_argument("--no-distances", action="store_true", help="Do NOT write the pairwise distances CSV.")
    p.add_argument("--no-inqscribe", action="store_true", help="Do NOT write the InqScribe file.")
    p.add_argument("--no-frames-tsv", action="store_true", help="Do NOT write the frame-only events TSV.")
    p.add_argument("--no-nans", action="store_true", help="Exclude NaN events.")
    p.add_argument("--anchor-frame", type=int, default=None, help="Anchor a specific frame to a wall time (e.g., 5826).")
    p.add_argument("--anchor-time", type=str, default=None, help="Wall time for the anchor frame (e.g., 0:03:14.000).")
    p.add_argument("--no-per-id-distance", action="store_true", help="Do NOT write the per-ID path length CSV.")
    p.add_argument("--no-trajectory-plot", action="store_true", help="Do NOT write the trajectory plot PDF.")

    args = p.parse_args()
    if (args.anchor_frame is None) ^ (args.anchor_time is None):
        p.error("--anchor-frame and --anchor-time must be provided together.")

    artifacts = process(
        csv_path=args.csv,
        output_prefix=args.output_prefix,
        fps=args.fps,
        threshold_px=args.threshold,
        min_frames=args.min_frames,
        export_distances=not args.no_distances,
        export_frames_only=not args.no_frames_tsv,
        export_inqscribe=not args.no_inqscribe,
        include_nan_events=not args.no_nans,
        export_per_id_distance=not args.no_per_id_distance,
        export_trajectory_plot=not args.no_trajectory_plot,
        anchor_frame=args.anchor_frame,
        anchor_time=args.anchor_time,
    )
    print("Generated:")
    for k, v in artifacts.items():
        print(f"  - {k}: {v}")
