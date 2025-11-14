#!/usr/bin/env python3
"""
IDTrackerAI_PostProcess_v1.py

GUI version of the IDTracker → InqScribe converter
with user-configurable:
- trajectories file
- ROI TOML file
- distance threshold (px)
- min contact duration (seconds)
- output directory (defaults to Post_Processing_Output next to the trajectories file, or inside selected base dir)
- optional base directory scan to auto-find trajectories.csv and .toml

Outputs:
1) <outdir>/<basename>_InqScribe_<thresh}px_<fps>fps.txt
2) <outdir>/<basename>_tracks_<thresh}px_<fps>fps.pdf  (multi-page, ROIs underneath)
3) <outdir>/<basename>_pairwise_distances_<thresh}px_<fps>fps.csv
4) <outdir>/<basename>_roi_events.csv              (if TOML provided)
5) <outdir>/<basename>_roi_summary.csv             (if TOML provided)
"""

import os
import re
import csv
import math
from pathlib import Path

# --- try tkinter, but fall back to CLI if missing ---
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    _HAS_TK = True
except Exception:
    _HAS_TK = False

import pandas as pd
import numpy as np
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ==============================
# HELPER FUNCTIONS
# ==============================
def fmt_hhmmss_comma_ms(seconds: float) -> str:
    """Convert seconds (float) → 'HH:MM:SS,mmm' (for InqScribe)."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    leftover = seconds % 60
    s_int = int(leftover)
    ms = int(round((leftover - s_int) * 1000))
    if ms == 1000:
        ms = 0
        s_int += 1
        if s_int == 60:
            s_int = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h:02d}:{m:02d}:{s_int:02d},{ms:03d}"


def segment_mask_frames(mask: np.ndarray, min_len_frames: int):
    """Return (start_frame, end_frame) tuples for True runs ≥ min_len_frames."""
    segments = []
    in_seg = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_seg:
            in_seg = True
            start = i
        elif not val and in_seg:
            end = i - 1
            if (end - start + 1) >= min_len_frames:
                segments.append((start, end))
            in_seg = False
    if in_seg:
        end = len(mask) - 1
        if (end - start + 1) >= min_len_frames:
            segments.append((start, end))
    return segments


def discover_ids(df: pd.DataFrame):
    """
    Find all xN/yN pairs, e.g. x1,y1 ... x12,y12.
    Return sorted list of ints.
    """
    ids = set()
    for col in df.columns:
        m = re.fullmatch(r"[xy](\d+)", col)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def make_tracks_pdf(df: pd.DataFrame, pdf_path: str, rois=None):
    """
    Create a multi-page PDF:
    - One page per individual ID (xN,yN)
    - One final page with all IDs together
    Colors are consistent across pages, and all pages use the same x/y limits.
    If ROIs are provided, draw them in black underneath on every page.
    """
    # find all IDs present
    ids = []
    for col in df.columns:
        m = re.fullmatch(r"x(\d+)", col)
        if m:
            idx = int(m.group(1))
            if f"y{idx}" in df.columns:
                ids.append(idx)
    ids = sorted(ids)
    if not ids:
        return

    # collect all x/y to get global extents
    all_x_list = []
    all_y_list = []
    for i in ids:
        x = df[f"x{i}"]
        y = df[f"y{i}"]
        all_x_list.append(x.to_numpy(dtype=float))
        all_y_list.append(y.to_numpy(dtype=float))
    all_x = np.concatenate([np.array(a, dtype=float) for a in all_x_list])
    all_y = np.concatenate([np.array(a, dtype=float) for a in all_y_list])
    all_x = all_x[~np.isnan(all_x)]
    all_y = all_y[~np.isnan(all_y)]
    if all_x.size > 0 and all_y.size > 0:
        xmin, xmax = all_x.min(), all_x.max()
        ymin, ymax = all_y.min(), all_y.max()
        pad_x = (xmax - xmin) * 0.05 if xmax > xmin else 10
        pad_y = (ymax - ymin) * 0.05 if ymax > ymin else 10
        xlim = (xmin - pad_x, xmax + pad_x)
        ylim = (ymin - pad_y, ymax + pad_y)
    else:
        xlim = None
        ylim = None

    # stable color map
    import itertools
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    color_cycle = itertools.cycle(prop_cycle)
    id_to_color = {i: next(color_cycle) for i in ids}

    with PdfPages(pdf_path) as pdf:
        # 1) individual pages
        for i in ids:
            x = df[f"x{i}"]
            y = df[f"y{i}"]
            fig, ax = plt.subplots(figsize=(6, 6))

            # draw rois first
            if rois:
                from matplotlib.patches import Polygon as _Polygon
                for roi in rois:
                    poly = roi["poly"]
                    ax.add_patch(_Polygon(poly, closed=True, fill=False,
                                          edgecolor="black", linewidth=1.0, alpha=0.6))

            color = id_to_color[i]
            ax.plot(x, y, '-', linewidth=1.4, alpha=0.9, label=f"ID{i}", color=color)

            # start / end markers
            valid = (~x.isna()) & (~y.isna())
            if valid.any():
                first_idx = valid.idxmax()
                ax.scatter([x.iloc[first_idx]], [y.iloc[first_idx]],
                           s=70, marker='^', color=color,
                           edgecolor='black', linewidth=0.5)
                last_idx = np.where(valid.values)[0][-1]
                ax.scatter([x.iloc[last_idx]], [y.iloc[last_idx]],
                           s=70, marker='o', color=color,
                           edgecolor='black', linewidth=0.5)

            if xlim is not None and ylim is not None:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
            ax.set_aspect('equal', adjustable='box')
            ax.invert_yaxis()
            ax.set_title(f"Tracked path – ID{i}")
            ax.set_xlabel("x (pixels)")
            ax.set_ylabel("y (pixels)")
            pdf.savefig(fig)
            plt.close(fig)

        # 2) combined page
        fig, ax = plt.subplots(figsize=(6, 6))

        # ROIs underneath
        if rois:
            from matplotlib.patches import Polygon as _Polygon
            for roi in rois:
                poly = roi["poly"]
                ax.add_patch(_Polygon(poly, closed=True, fill=False,
                                      edgecolor="black", linewidth=1.0, alpha=0.6))

        line_handles = []
        for i in ids:
            x = df[f"x{i}"]
            y = df[f"y{i}"]
            color = id_to_color[i]
            (line,) = ax.plot(x, y, '-', linewidth=1.2, alpha=0.9, label=f"ID{i}", color=color)
            line_handles.append(line)

        if xlim is not None and ylim is not None:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        ax.set_title("Tracked paths (all IDs)")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.legend(handles=line_handles, loc="lower left", frameon=False)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def write_pairwise_distances(df: pd.DataFrame, ids, base: str,
                             fps: float, threshold_px: float) -> str:
    """
    For every pair of IDs (i,j), write frame-by-frame distances.
    """
    out_path = f"{base}_pairwise_distances_{int(threshold_px)}px_{fps:.3f}fps.csv"

    num_frames = len(df)
    frames = np.arange(num_frames, dtype=int)
    time_s = frames / fps

    coord_cache = {}
    valid_cache = {}
    for i in ids:
        xi = df.get(f"x{i}")
        yi = df.get(f"y{i}")
        if xi is None or yi is None:
            continue
        xi_vals = xi.to_numpy(dtype=float, copy=False)
        yi_vals = yi.to_numpy(dtype=float, copy=False)
        coord_cache[i] = (xi_vals, yi_vals)
        valid_cache[i] = ~(np.isnan(xi_vals) | np.isnan(yi_vals))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame",
            "time_s",
            "id_i",
            "id_j",
            "distance_px",
            "both_valid",
            f"contact_le_{int(threshold_px)}px",
        ])

        for idx_i in range(len(ids)):
            i = ids[idx_i]
            if i not in coord_cache:
                continue
            xi_vals, yi_vals = coord_cache[i]
            valid_i = valid_cache[i]
            for idx_j in range(idx_i + 1, len(ids)):
                j = ids[idx_j]
                if j not in coord_cache:
                    continue
                xj_vals, yj_vals = coord_cache[j]
                valid_j = valid_cache[j]

                both_valid = valid_i & valid_j
                distance = np.hypot(xi_vals - xj_vals, yi_vals - yj_vals)
                contact = both_valid & (distance <= threshold_px)

                w.writerows(
                    (
                        frame,
                        time,
                        i,
                        j,
                        "" if (not bv or np.isnan(dist)) else float(dist),
                        int(bv),
                        int(ct),
                    )
                    for frame, time, dist, bv, ct in
                    zip(frames, time_s, distance, both_valid, contact)
                )

    return out_path


# ==============================
# ROI HELPER FUNCTIONS
# ==============================
def parse_roi_string(s: str):
    m = re.search(r"\[\s*\[.*\]\s*\]", s)
    if not m:
        raise ValueError(f"Could not parse ROI polygon from: {s}")
    coords = eval(m.group(0), {"__builtins__": {}})
    return [(float(x), float(y)) for (x, y) in coords]


def load_rois_from_toml(toml_path: str):
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    roi_list = data.get("roi_list")
    if not roi_list:
        raise ValueError("No 'roi_list' found in TOML")
    rois = []
    for idx, roi_str in enumerate(roi_list):
        poly = parse_roi_string(roi_str)
        if idx == 0:
            name = "arena"
        else:
            name = f"bracket{idx}"
        rois.append({"name": name, "poly": poly})
    return rois


def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y))
        if cond:
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x <= xinters:
                inside = not inside
    return inside


def write_roi_outputs(df: pd.DataFrame, rois, beetle_ids, fps: float, out_base: str):
    num_frames = len(df)
    prev_inside = {bid: {roi["name"]: False for roi in rois} for bid in beetle_ids}
    frames_in_roi = {bid: {roi["name"]: 0 for roi in rois} for bid in beetle_ids}
    dist_in_roi = {bid: {roi["name"]: 0.0 for roi in rois} for bid in beetle_ids}
    prev_pos = {bid: (math.nan, math.nan) for bid in beetle_ids}
    events = []

    for frame_idx, row in df.iterrows():
        t_s = frame_idx / fps
        for bid in beetle_ids:
            x = row[f"x{bid}"]
            y = row[f"y{bid}"]
            valid = not (math.isnan(x) or math.isnan(y))
            for roi in rois:
                roi_name = roi["name"]
                if valid:
                    inside = point_in_poly(x, y, roi["poly"])
                else:
                    inside = False
                was_inside = prev_inside[bid][roi_name]
                if inside:
                    frames_in_roi[bid][roi_name] += 1
                if inside and not was_inside:
                    events.append({
                        "frame": frame_idx,
                        "time_s": t_s,
                        "beetle_id": bid,
                        "roi_name": roi_name,
                        "event_type": "ENTER",
                    })
                elif (not inside) and was_inside:
                    events.append({
                        "frame": frame_idx,
                        "time_s": t_s,
                        "beetle_id": bid,
                        "roi_name": roi_name,
                        "event_type": "EXIT",
                    })
                prev_inside[bid][roi_name] = inside

            # distance inside rois
            prev_x, prev_y = prev_pos[bid]
            if valid and not (math.isnan(prev_x) or math.isnan(prev_y)):
                step = math.hypot(x - prev_x, y - prev_y)
                for roi in rois:
                    roi_name = roi["name"]
                    if prev_inside[bid][roi_name]:
                        dist_in_roi[bid][roi_name] += step
            prev_pos[bid] = (x, y)

    events_path = f"{out_base}_roi_events.csv"
    with open(events_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time_s", "beetle_id", "roi_name", "event_type"])
        w.writeheader()
        for ev in events:
            w.writerow(ev)

    summary_path = f"{out_base}_roi_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "beetle_id",
            "roi_name",
            "frames_in_roi",
            "pct_time_in_roi",
            "dist_in_roi_px",
            "total_frames",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for bid in beetle_ids:
            for roi in rois:
                roi_name = roi["name"]
                fin = frames_in_roi[bid][roi_name]
                pct = fin / num_frames if num_frames else 0.0
                w.writerow({
                    "beetle_id": bid,
                    "roi_name": roi_name,
                    "frames_in_roi": fin,
                    "pct_time_in_roi": pct,
                    "dist_in_roi_px": dist_in_roi[bid][roi_name],
                    "total_frames": num_frames,
                })

    return events_path, summary_path


def process_trajectories(csv_path: str,
                         fps: float,
                         threshold_px: float,
                         min_frames: int,
                         output_dir: str,
                         toml_path: str | None = None):
    """
    Core logic. All configurable values passed in.
    """
    df = pd.read_csv(csv_path)

    ids = discover_ids(df)
    if not ids:
        raise ValueError("No xN/yN columns found in this file.")

    num_frames = len(df)
    frames = np.arange(num_frames, dtype=int)
    time_seconds = frames / fps

    # per-ID cache
    per_id = {}
    for i in ids:
        xi = df[f"x{i}"]
        yi = df[f"y{i}"]
        valid = ~(xi.isna() | yi.isna())
        per_id[i] = {"x": xi, "y": yi, "valid": valid}

    # build events for all pairs + NaNs
    events = []
    for idx_i in range(len(ids)):
        i = ids[idx_i]
        for idx_j in range(idx_i + 1, len(ids)):
            j = ids[idx_j]
            xi = per_id[i]["x"]
            yi = per_id[i]["y"]
            xj = per_id[j]["x"]
            yj = per_id[j]["y"]
            valid_i = per_id[i]["valid"]
            valid_j = per_id[j]["valid"]
            both_valid = valid_i & valid_j
            distance_px = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            proximity_mask = (distance_px <= threshold_px) & both_valid
            proximity_segments = segment_mask_frames(proximity_mask.values, min_frames)
            for s, e in proximity_segments:
                events.append(
                    (
                        time_seconds[s],
                        time_seconds[e],
                        f"Proximity: ID{i}-ID{j} [F{s}–F{e}]",
                        f"distance <= {int(threshold_px)}px; duration={e - s + 1} frames; IDs {i},{j} valid",
                    )
                )

    # NaNs
    for i in ids:
        valid_i = per_id[i]["valid"]
        nan_mask = ~valid_i.values
        nan_segments = segment_mask_frames(nan_mask, min_frames)
        for s, e in nan_segments:
            events.append(
                (
                    time_seconds[s],
                    time_seconds[e],
                    f"NaN: ID{i} missing [F{s}–F{e}]",
                    f"ID{i} missing for {e - s + 1} frames",
                )
            )

    events.sort(key=lambda r: (r[0], r[1], r[2]))

    # base name in output dir
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    base = os.path.join(output_dir, base_name)

    # write InqScribe (always)
    inq_path = f"{base}_InqScribe_{int(threshold_px)}px_{fps:.0f}fps.txt"
    with open(inq_path, "w", encoding="utf-8-sig", newline="\n") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["Start Time", "End Time", "Title", "Comment"])
        for start_s, end_s, title, comment in events:
            w.writerow([
                fmt_hhmmss_comma_ms(start_s),
                fmt_hhmmss_comma_ms(end_s),
                title,
                comment,
            ])

    # load rois (if any)
    rois = None
    if toml_path is not None and toml_path != "":
        rois = load_rois_from_toml(toml_path)

    # PDF
    pdf_path = f"{base}_tracks_{int(threshold_px)}px_{fps:.0f}fps.pdf"
    make_tracks_pdf(df, pdf_path, rois if rois else None)

    # pairwise
    pairwise_path = write_pairwise_distances(df, ids, base, fps, threshold_px)

    # ROI CSVs
    roi_events_path = None
    roi_summary_path = None
    if rois is not None:
        base_for_roi = os.path.join(output_dir, base_name)
        roi_events_path, roi_summary_path = write_roi_outputs(df, rois, ids, fps, base_for_roi)

    return inq_path, pdf_path, pairwise_path, roi_events_path, roi_summary_path, len(events)


# ==============================
# GUI PART
# ==============================
def run_gui():
    root = tk.Tk()
    root.title("IDTrackerAI Post-Process")

    # vars
    csv_var = tk.StringVar()
    toml_var = tk.StringVar()
    basedir_var = tk.StringVar()
    thresh_var = tk.StringVar(value="60")
    dur_var = tk.StringVar(value="0.2")  # seconds
    outdir_var = tk.StringVar()
    frames_hint_var = tk.StringVar(value="≈ 6 frames at 30 fps")

    def update_frames_hint(event=None):
        try:
            dur_seconds = float(dur_var.get().strip())
            fps = 30.0
            frames = max(1, int(round(dur_seconds * fps)))
            frames_hint_var.set(f"≈ {frames} frames at {int(fps)} fps")
        except ValueError:
            frames_hint_var.set("")

    def choose_basedir():
        d = filedialog.askdirectory(title="Select base directory to scan")
        if not d:
            return
        basedir_var.set(d)

        # scan recursively for trajectories.csv and .toml
        traj_files = []
        toml_files = []
        for rootdir, _dirs, files in os.walk(d):
            for fname in files:
                lower = fname.lower()
                full = os.path.join(rootdir, fname)
                if lower == "trajectories.csv":
                    traj_files.append(full)
                elif lower.endswith(".toml"):
                    toml_files.append(full)

        if len(traj_files) == 1:
            csv_var.set(traj_files[0])
        elif len(traj_files) > 1:
            messagebox.showwarning(
                "Multiple trajectories.csv found",
                "More than one trajectories.csv was found in this directory tree. Please pick the correct one manually.",
            )

        if len(toml_files) == 1:
            toml_var.set(toml_files[0])
        elif len(toml_files) > 1:
            messagebox.showwarning(
                "Multiple TOML files found",
                "More than one .toml file was found in this directory tree. Please pick the correct one manually.",
            )

        # default output dir = Post_Processing_Output inside chosen dir
        ppo = os.path.join(d, "Post_Processing_Output")
        outdir_var.set(ppo)

    def choose_csv():
        path = filedialog.askopenfilename(
            title="Select trajectories.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            csv_var.set(path)
            csv_dir = os.path.dirname(path)
            ppo = os.path.join(csv_dir, "Post_Processing_Output")
            outdir_var.set(ppo)

    def choose_outdir():
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            outdir_var.set(d)

    def do_run():
        csv_path = csv_var.get().strip()
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("Error", "Please choose a valid trajectories.csv")
            return

        # output dir: if blank, make Post_Processing_Output next to CSV
        out_dir = outdir_var.get().strip()
        if not out_dir:
            csv_dir = os.path.dirname(csv_path)
            out_dir = os.path.join(csv_dir, "Post_Processing_Output")
        os.makedirs(out_dir, exist_ok=True)

        # parse numeric inputs
        try:
            threshold_px = float(thresh_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Distance threshold must be a number.")
            return

        try:
            dur_seconds = float(dur_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Duration must be a number (seconds).")
            return

        fps = 30.0  # could be made user-configurable
        min_frames = max(1, int(round(dur_seconds * fps)))
        toml_path = toml_var.get().strip()

        try:
            inq_path, pdf_path, pairwise_path, roi_events_path, roi_summary_path, n_events = process_trajectories(
                csv_path,
                fps=fps,
                threshold_px=threshold_px,
                min_frames=min_frames,
                output_dir=out_dir,
                toml_path=toml_path,
            )
            msg = f"Created:\n{inq_path}\n{pdf_path}\n{pairwise_path}\n"
            if roi_events_path:
                msg += f"{roi_events_path}\n{roi_summary_path}\n"
            msg += f"\nEvents: {n_events}"
            messagebox.showinfo("Success", msg)
            root.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print("ERROR:", e)

    # layout
    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill="both", expand=True)

    # base directory row
    tk.Label(frm, text="Base directory:").grid(row=0, column=0, sticky="w")
    tk.Entry(frm, textvariable=basedir_var, width=50).grid(row=0, column=1, sticky="we")
    tk.Button(frm, text="Browse…", command=choose_basedir).grid(row=0, column=2, padx=5)

    # CSV
    tk.Label(frm, text="Trajectories CSV:").grid(row=1, column=0, sticky="w")
    tk.Entry(frm, textvariable=csv_var, width=50).grid(row=1, column=1, sticky="we")
    tk.Button(frm, text="Browse…", command=choose_csv).grid(row=1, column=2, padx=5)

    # ROI TOML
    tk.Label(frm, text="ROI TOML file:").grid(row=2, column=0, sticky="w")
    tk.Entry(frm, textvariable=toml_var, width=50).grid(row=2, column=1, sticky="we")
    tk.Button(
        frm,
        text="Browse…",
        command=lambda: toml_var.set(
            filedialog.askopenfilename(
                title="Select ROI TOML",
                filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            )
        ),
    ).grid(row=2, column=2, padx=5)

    # threshold
    tk.Label(frm, text="Distance threshold (px):").grid(row=3, column=0, sticky="w")
    tk.Entry(frm, textvariable=thresh_var, width=10).grid(row=3, column=1, sticky="w")

    # duration
    tk.Label(frm, text="Min contact duration (s):").grid(row=4, column=0, sticky="w")
    dur_entry = tk.Entry(frm, textvariable=dur_var, width=10)
    dur_entry.grid(row=4, column=1, sticky="w")
    dur_entry.bind("<FocusOut>", update_frames_hint)
    tk.Label(frm, textvariable=frames_hint_var, fg="#555").grid(row=4, column=2, sticky="w")

    # output dir
    tk.Label(frm, text="Output directory:").grid(row=5, column=0, sticky="w")
    tk.Entry(frm, textvariable=outdir_var, width=50).grid(row=5, column=1, sticky="we")
    tk.Button(frm, text="Choose…", command=choose_outdir).grid(row=5, column=2, padx=5)

    # run button
    tk.Button(
        frm,
        text="Run",
        width=12,
        command=do_run
    ).grid(row=6, column=0, columnspan=3, pady=10, sticky="we")

    frm.columnconfigure(1, weight=1)

    root.mainloop()


# ==============================
# CLI FALLBACK
# ==============================
def main():
    if _HAS_TK:
        run_gui()
    else:
        print("Tkinter not available — CLI fallback.")
        csv_path = input("Path to trajectories.csv: ").strip()
        if not csv_path or not os.path.exists(csv_path):
            print("File not found, exiting.")
            return
        toml_path = input("Path to ROI TOML (or leave blank): ").strip()
        out_dir = os.path.join(os.path.dirname(csv_path), "Post_Processing_Output")
        os.makedirs(out_dir, exist_ok=True)
        try:
            inq_path, pdf_path, pairwise_path, roi_events_path, roi_summary_path, n_events = process_trajectories(
                csv_path,
                fps=30.0,
                threshold_px=60.0,
                min_frames=int(round(0.2 * 30.0)),
                output_dir=out_dir,
                toml_path=toml_path,
            )
            print("Created:")
            print(inq_path)
            print(pdf_path)
            print(pairwise_path)
            if roi_events_path:
                print(roi_events_path)
                print(roi_summary_path)
            print("Events:", n_events)
        except Exception as e:
            print("ERROR:", e)


if __name__ == "__main__":
    main()