#!/usr/bin/env python3
"""
idtracker_to_inqscribe_gui.py

GUI version of the IDTracker → InqScribe converter.

Features:
- Tkinter file picker to choose trajectories.csv
- Detects proximity events for ID1–ID2 (≤ 60 px for ≥ 6 frames) at 30 fps
- Detects NaN segments (both/ID1/ID2) if enabled
- Writes:
    1) <file>_InqScribe_60px_30fps.txt   — InqScribe import (for ID1–ID2)
    2) <file>_QA_by_frame.csv            — per-frame info for ID1–ID2
    3) <file>_tracks.pdf                 — path plot, start=triangle, end=circle
    4) <file>_pairwise_distances_60px_30fps.csv — NEW: all pairwise distances
       for every detected individual (up to 12)

We auto-discover individuals by looking for columns named xN / yN.
"""

import os
import re
import csv
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================
# CONFIGURABLE PARAMETERS
# ==============================
FPS = 30.0           # frames per second
THRESHOLD_PX = 60.0  # proximity threshold in pixels
MIN_FRAMES = 6       # minimum consecutive frames for an event
INCLUDE_NANS = True  # flag NaN segments


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
    Look through columns and find all xN/yN pairs, e.g. x1,y1 ... x12,y12.
    Return a sorted list of integers: [1,2,3,...].
    """
    ids = set()
    for col in df.columns:
        m = re.fullmatch(r"[xy](\d+)", col)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def make_tracks_pdf(df: pd.DataFrame, pdf_path: str):
    """
    Draw both trajectories for ID1 and ID2 (if present) and mark:
    - triangle (^) = first valid point
    - circle (o)   = last valid point
    Markers use same color as trajectory lines.
    Legend shows only line labels.
    If you have more than 2 IDs, we still only draw 1 and 2 here (matches your workflow).
    """
    if not {"x1", "y1", "x2", "y2"}.issubset(df.columns):
        # nothing to draw
        return

    x1, y1 = df["x1"], df["y1"]
    x2, y2 = df["x2"], df["y2"]

    plt.figure(figsize=(6, 6))

    # Draw trajectories
    line1, = plt.plot(x1, y1, "-", linewidth=1.5, alpha=0.9, label="ID1")
    line2, = plt.plot(x2, y2, "-", linewidth=1.5, alpha=0.9, label="ID2")

    color1 = line1.get_color()
    color2 = line2.get_color()

    # ---- ID1 start & end ----
    valid_id1 = (~x1.isna()) & (~y1.isna())
    if valid_id1.any():
        first_id1 = valid_id1.idxmax()
        plt.scatter([x1.iloc[first_id1]], [y1.iloc[first_id1]],
                    s=70, marker="^", color=color1, edgecolor="black", linewidth=0.5)
        last_id1 = np.where(valid_id1.values)[0][-1]
        plt.scatter([x1.iloc[last_id1]], [y1.iloc[last_id1]],
                    s=70, marker="o", color=color1, edgecolor="black", linewidth=0.5)

    # ---- ID2 start & end ----
    valid_id2 = (~x2.isna()) & (~y2.isna())
    if valid_id2.any():
        first_id2 = valid_id2.idxmax()
        plt.scatter([x2.iloc[first_id2]], [y2.iloc[first_id2]],
                    s=70, marker="^", color=color2, edgecolor="black", linewidth=0.5)
        last_id2 = np.where(valid_id2.values)[0][-1]
        plt.scatter([x2.iloc[last_id2]], [y2.iloc[last_id2]],
                    s=70, marker="o", color=color2, edgecolor="black", linewidth=0.5)

    plt.gca().invert_yaxis()
    plt.title("Tracked paths (ID1 vs ID2)")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.legend(handles=[line1, line2], loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()


def write_pairwise_distances(df: pd.DataFrame, ids, base: str, fps: float, threshold_px: float):
    """
    NEW:
    For every pair of IDs (i,j) where i < j, compute distance for every frame.
    Write to a CSV:
        frame, time_s, id_i, id_j, distance_px, both_valid, contact_le_thresh
    """
    out_path = f"{base}_pairwise_distances_{int(threshold_px)}px_{fps:.3f}fps.csv"

    num_frames = len(df)
    frames = np.arange(num_frames, dtype=int)
    time_s = frames / fps

    # Convert coordinate columns once so we can stream rows per pair without
    # re-reading pandas Series or building large Python lists.
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
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "time_s",
            "id_i",
            "id_j",
            "distance_px",
            "both_valid",
            f"contact_le_{int(threshold_px)}px",
        ])

        # loop over all pairs
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

                writer.writerows(
                    (
                        frame,
                        time,
                        i,
                        j,
                        "" if (not bv or np.isnan(dist)) else float(dist),
                        int(bv),
                        int(ct),
                    )
                    for frame, time, dist, bv, ct in zip(frames, time_s, distance, both_valid, contact)
                )

    return out_path


def process_trajectories(csv_path: str):
    """Core logic: read trajectories, detect events, write outputs."""
    df = pd.read_csv(csv_path)

    # discover all IDs present (x1,y1 ... x12,y12)
    ids = discover_ids(df)
    if not ids:
        raise ValueError("No xN/yN columns found in this file.")

    num_frames = len(df)
    frames = np.arange(num_frames, dtype=int)
    time_seconds = frames / FPS

    # We'll keep the original "ID1 vs ID2" event logic as your main use case.
    has_id1_id2 = {1, 2}.issubset(set(ids))

    if has_id1_id2:
        x1, y1 = df["x1"], df["y1"]
        x2, y2 = df["x2"], df["y2"]

        valid1 = ~(x1.isna() | y1.isna())
        valid2 = ~(x2.isna() | y2.isna())
        both_valid = valid1 & valid2

        distance_px = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        proximity_mask = (distance_px <= THRESHOLD_PX) & both_valid
        proximity_segments = segment_mask_frames(proximity_mask.values, MIN_FRAMES)

        nan1_mask = ~valid1.values
        nan2_mask = ~valid2.values
        both_nan_mask = nan1_mask & nan2_mask
        id1_only_mask = nan1_mask & ~both_nan_mask
        id2_only_mask = nan2_mask & ~both_nan_mask

        both_nan_segments = segment_mask_frames(both_nan_mask, MIN_FRAMES)
        id1_nan_segments = segment_mask_frames(id1_only_mask, MIN_FRAMES)
        id2_nan_segments = segment_mask_frames(id2_only_mask, MIN_FRAMES)

        events = []

        # Proximity events (ID1–ID2)
        for s, e in proximity_segments:
            events.append(
                (
                    time_seconds[s],
                    time_seconds[e],
                    f"Proximity: ID1-ID2 [F{s}–F{e}]",
                    f"distance <= {int(THRESHOLD_PX)}px; duration={e - s + 1} frames; both valid",
                )
            )

        # NaN events
        if INCLUDE_NANS:
            for s, e in both_nan_segments:
                events.append(
                    (
                        time_seconds[s],
                        time_seconds[e],
                        f"NaN: both IDs missing [F{s}–F{e}]",
                        f"duration={e - s + 1} frames",
                    )
                )
            for s, e in id1_nan_segments:
                events.append(
                    (
                        time_seconds[s],
                        time_seconds[e],
                        f"NaN: ID1 missing [F{s}–F{e}]",
                        f"duration={e - s + 1} frames",
                    )
                )
            for s, e in id2_nan_segments:
                events.append(
                    (
                        time_seconds[s],
                        time_seconds[e],
                        f"NaN: ID2 missing [F{s}–F{e}]",
                        f"duration={e - s + 1} frames",
                    )
                )

        events.sort(key=lambda r: (r[0], r[1], r[2]))
    else:
        # no id1/id2 -> still want to write pairwise, but skip inqscribe
        events = []
        both_valid = pd.Series([False] * num_frames)
        distance_px = pd.Series([np.nan] * num_frames)

    base, _ = os.path.splitext(csv_path)
    inq_path = f"{base}_InqScribe_60px_30fps.txt"
    qa_path = f"{base}_QA_by_frame.csv"
    pdf_path = f"{base}_tracks.pdf"

    # write InqScribe only if we have ID1 & ID2
    if has_id1_id2:
        with open(inq_path, "w", encoding="utf-8-sig", newline="\n") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(["Start Time", "End Time", "Title", "Comment"])
            for start_s, end_s, title, comment in events:
                writer.writerow([
                    fmt_hhmmss_comma_ms(start_s),
                    fmt_hhmmss_comma_ms(end_s),
                    title,
                    comment,
                ])
    else:
        inq_path = "(no ID1/ID2, InqScribe not written)"

    # write QA for ID1–ID2 if present
    if has_id1_id2:
        qa_df = pd.DataFrame({
            "frame": frames,
            "time_s": time_seconds,
            "distance_px": distance_px,
            "both_valid": both_valid.astype(int),
            "contact_le_60px": ((distance_px <= THRESHOLD_PX) & both_valid).astype(int),
        })
        qa_df.to_csv(qa_path, index=False)
    else:
        qa_path = "(no ID1/ID2, QA not written)"

    # write PDF of tracks (only for ID1–ID2)
    if has_id1_id2:
        make_tracks_pdf(df, pdf_path)
    else:
        pdf_path = "(no ID1/ID2, PDF not written)"

    # NEW: write pairwise distances for all detected IDs
    pairwise_path = write_pairwise_distances(df, ids, base, FPS, THRESHOLD_PX)

    return inq_path, qa_path, pdf_path, pairwise_path, len(events)


def main():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Select file", "Pick your IDTracker trajectories.csv")
    csv_path = filedialog.askopenfilename(
        title="Select trajectories.csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    if not csv_path:
        messagebox.showwarning("No file", "No file selected.")
        return

    try:
        inq_path, qa_path, pdf_path, pairwise_path, n_events = process_trajectories(csv_path)
        messagebox.showinfo(
            "Done",
            f"Created:\n{inq_path}\n{qa_path}\n{pdf_path}\n{pairwise_path}\n\n"
            f"ID1–ID2 events: {n_events}",
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    main()
