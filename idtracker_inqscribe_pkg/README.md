# idtracker-inqscribe

Convert **IDTracker** trajectory outputs into:
- a **strict InqScribe** import file (UTF-8 BOM; `HH:MM:SS,mmm`),
- a **frame-only** TSV (no timecodes) for QA/reproducibility,
- an optional **pairwise distances** CSV,
- a per-beetle **path length summary** (total distance travelled per ID),
- an optional **trajectory plot PDF** that traces each beetle.

This tool is **frame-first**: it uses *frame indices* as ground truth and renders time as `frame / FPS` (default 30.000).

## Install (local file)
```
pip install idtracker_inqscribe-0.1.0.zip
```
(or `pip install .` from the unpacked folder)

## CLI
```
idtracker-to-inqscribe /path/to/trajectories.csv   --fps 30   --threshold 60   --min-frames 6   --anchor-frame 5826 --anchor-time 0:03:14.000
```

See `idtracker-to-inqscribe --help` for all options.

Launch the GUI instead by running `idtracker-to-inqscribe` with no arguments; select the trajectories CSV, distance threshold, and minimum proximity duration (in frames), then click **Run**.
