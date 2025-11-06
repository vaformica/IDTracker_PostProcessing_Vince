#!/usr/bin/env python3
"""
IDTrackerAI_PostProcess_v1.py

GUI version of the IDTracker → InqScribe converter
with user-configurable:
- trajectories file
- distance threshold (px)
- min contact duration (seconds)
- output directory (defaults to directory of trajectories file)

Outputs:
1) <outdir>/<basename>_InqScribe_<thresh}px_<fps>fps.txt
2) <outdir>/<basename>_tracks.pdf
3) <outdir>/<basename>_pairwise_distances_<thresh}px_<fps>fps.csv
"""

import os
import re
import csv

# --- try tkinter, but fall back to CLI if missing ---
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    _HAS_TK = True
except Exception:
    _HAS_TK = False

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def make_tracks_pdf(df: pd.DataFrame, pdf_path: str):
    """
    Draw trajectories for all detected IDs (x1,y1 ... x12,y12).
    - Plots every ID's path
    - Still marks start/end for ID1 and ID2 (your primary pair)
    - Legend in lower-left so it doesn't occlude
    """
    # find all IDs present
    ids = []
    for col in df.columns:
        # look for xN / yN pattern
        import re as _re
        m = _re.fullmatch(r"x(\d+)", col)
        if m:
            idx = int(m.group(1))
            if f"y{idx}" in df.columns:
                ids.append(idx)
    ids = sorted(ids)
    if not ids:
        return

    plt.figure(figsize=(6, 6))
    line_handles = []

    # plot every ID we found
    for i in ids:
        x = df[f"x{i}"]
        y = df[f"y{i}"]
        (line,) = plt.plot(x, y, '-', linewidth=1.2, alpha=0.9, label=f"ID{i}")
        line_handles.append(line)

    # mark starts/ends for ID1 and ID2 if present
    for i, color_line in [(1, None), (2, None)]:
        if f"x{i}" in df.columns and f"y{i}" in df.columns:
            x = df[f"x{i}"]
            y = df[f"y{i}"]
            valid = (~x.isna()) & (~y.isna())
            if valid.any():
                # get the line color we used above
                # find the corresponding handle
                handle = None
                for h in line_handles:
                    if h.get_label() == f"ID{i}":
                        handle = h
                        break
                color = handle.get_color() if handle is not None else None

                first_idx = valid.idxmax()
                plt.scatter([x.iloc[first_idx]], [y.iloc[first_idx]],
                            s=70, marker='^', color=color, edgecolor='black', linewidth=0.5)
                last_idx = np.where(valid.values)[0][-1]
                plt.scatter([x.iloc[last_idx]], [y.iloc[last_idx]],
                            s=70, marker='o', color=color, edgecolor='black', linewidth=0.5)

    plt.gca().invert_yaxis()
    plt.title("Tracked paths (all IDs)")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.legend(handles=line_handles, loc="lower left", frameon=False)
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()


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


def process_trajectories(csv_path: str,
                         fps: float,
                         threshold_px: float,
                         min_frames: int,
                         output_dir: str):
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

    has_id1_id2 = {1, 2}.issubset(set(ids))

    if has_id1_id2:
        x1, y1 = df["x1"], df["y1"]
        x2, y2 = df["x2"], df["y2"]

        valid1 = ~(x1.isna() | y1.isna())
        valid2 = ~(x2.isna() | y2.isna())
        both_valid = valid1 & valid2

        distance_px = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        proximity_mask = (distance_px <= threshold_px) & both_valid
        proximity_segments = segment_mask_frames(proximity_mask.values, min_frames)

        # NaNs
        nan1_mask = ~valid1.values
        nan2_mask = ~valid2.values
        both_nan_mask = nan1_mask & nan2_mask
        id1_only_mask = nan1_mask & ~both_nan_mask
        id2_only_mask = nan2_mask & ~both_nan_mask

        both_nan_segments = segment_mask_frames(both_nan_mask, min_frames)
        id1_nan_segments = segment_mask_frames(id1_only_mask, min_frames)
        id2_nan_segments = segment_mask_frames(id2_only_mask, min_frames)

        events = []
        for s, e in proximity_segments:
            events.append(
                (
                    time_seconds[s],
                    time_seconds[e],
                    f"Proximity: ID1-ID2 [F{s}–F{e}]",
                    f"distance <= {int(threshold_px)}px; duration={e - s + 1} frames; both valid",
                )
            )
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
        events = []

    # build base name in output dir
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    base = os.path.join(output_dir, base_name)

    # InqScribe
    if has_id1_id2:
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
    else:
        inq_path = "(no ID1/ID2, InqScribe not written)"

    # PDF
    if has_id1_id2:
        pdf_path = f"{base}_tracks_{int(threshold_px)}px_{fps:.0f}fps.pdf"
        make_tracks_pdf(df, pdf_path)
    else:
        pdf_path = "(no ID1/ID2, PDF not written)"

    # Pairwise
    pairwise_path = write_pairwise_distances(df, ids, base, fps, threshold_px)

    return inq_path, pdf_path, pairwise_path, len(events)


# ==============================
# GUI PART
# ==============================
def run_gui():
    root = tk.Tk()
    root.title("IDTrackerAI Post-Process")

    # vars
    csv_var = tk.StringVar()
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

    def choose_csv():
        path = filedialog.askopenfilename(
            title="Select trajectories.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            csv_var.set(path)
            # default output dir to CSV dir
            outdir_var.set(os.path.dirname(path))

    def choose_outdir():
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            outdir_var.set(d)

    def do_run():
        csv_path = csv_var.get().strip()
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("Error", "Please choose a valid trajectories.csv")
            return
        out_dir = outdir_var.get().strip() or os.path.dirname(csv_path)
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

        fps = 30.0  # fixed, you can add a box later
        min_frames = max(1, int(round(dur_seconds * fps)))

        try:
            inq_path, pdf_path, pairwise_path, n_events = process_trajectories(
                csv_path,
                fps=fps,
                threshold_px=threshold_px,
                min_frames=min_frames,
                output_dir=out_dir,
            )
            messagebox.showinfo(
                "Success",
                f"Created:\n{inq_path}\n{pdf_path}\n{pairwise_path}\n\n"
                f"ID1–ID2 events: {n_events}",
            )
            root.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print("ERROR:", e)

    # layout
    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill="both", expand=True)

    # CSV
    tk.Label(frm, text="Trajectories CSV:").grid(row=0, column=0, sticky="w")
    tk.Entry(frm, textvariable=csv_var, width=50).grid(row=0, column=1, sticky="we")
    tk.Button(frm, text="Browse…", command=choose_csv).grid(row=0, column=2, padx=5)

    # threshold
    tk.Label(frm, text="Distance threshold (px):").grid(row=1, column=0, sticky="w")
    tk.Entry(frm, textvariable=thresh_var, width=10).grid(row=1, column=1, sticky="w")

    # duration
    tk.Label(frm, text="Min contact duration (s):").grid(row=2, column=0, sticky="w")
    dur_entry = tk.Entry(frm, textvariable=dur_var, width=10)
    dur_entry.grid(row=2, column=1, sticky="w")
    dur_entry.bind("<FocusOut>", update_frames_hint)
    tk.Label(frm, textvariable=frames_hint_var, fg="#555").grid(row=2, column=2, sticky="w")

    # output dir
    tk.Label(frm, text="Output directory:").grid(row=3, column=0, sticky="w")
    tk.Entry(frm, textvariable=outdir_var, width=50).grid(row=3, column=1, sticky="we")
    tk.Button(frm, text="Choose…", command=choose_outdir).grid(row=3, column=2, padx=5)

    # run button
    tk.Button(
        frm,
        text="Run",
        width=12,
        command=do_run
    ).grid(row=4, column=0, columnspan=3, pady=10, sticky="we")

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
        out_dir = os.path.dirname(csv_path)
        try:
            inq_path, pdf_path, pairwise_path, n_events = process_trajectories(
                csv_path,
                fps=30.0,
                threshold_px=60.0,
                min_frames=int(round(0.2 * 30.0)),
                output_dir=out_dir,
            )
            print("Created:")
            print(inq_path)
            print(pdf_path)
            print(pairwise_path)
            print("ID1–ID2 events:", n_events)
        except Exception as e:
            print("ERROR:", e)


if __name__ == "__main__":
    main()