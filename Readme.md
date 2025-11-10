# IDTrackerAI Post‑Processing GUI

## Overview

This GUI allows you to load the outputs from IDTrackerAI and manually correct any misidentifications or errors in the tracking data. It is designed to streamline the post-processing of animal tracking data by providing an intuitive interface to review and edit the results.

## What the Script Produces

Before running the script, ensure your selected directory is the same folder produced by IDTracker.ai for a specific video. This folder must include:
- `trajectories.csv` — The tracked coordinates of all individuals across frames.
- `.toml` — The ROI definitions describing the arena and any fungus brackets.
These two files are essential; the script will not proceed without them.

- A **pairwise distances CSV** file containing distances between all tracked beetles across all frames. This allows further quantitative analysis of proximity patterns and interactions over time.

- An **InqScribe‑ready tab‑delimited text file** that can be imported directly into InqScribe. This file now includes proximity events for all pairs of beetles (not just ID1 and ID2), as well as periods where any beetle’s position data are missing (NAs). These NAs often indicate fights or overlapping individuals where IDTracker.ai temporarily lost track of one or both beetles.

- A **multi-page PDF visualization** showing the tracked paths of each beetle. Each beetle has its own page with its individual trajectory (color-coded consistently across pages), and the final page shows all beetles combined. The axes are scaled equally and share the same limits across all pages to preserve spatial proportions. Black polygons show the ROIs (the arena and individual fungus brackets).
  ![Example tracked paths](./example_tracks_output.png)
  *Figure: Example of the combined page in the output PDF. Each beetle’s trajectory is plotted in a unique color, with triangles marking starting positions and circles marking ending positions.*

All outputs are saved in the `Post_Processing_Output` directory by default.

## How It Works

1. Load the IDTrackerAI output files into the GUI.
2. Visualize the tracking data frame-by-frame.
3. Use the GUI tools to identify and correct mislabelled individuals.
4. Save the corrected data for further analysis.

5. The script automatically computes pairwise distances for all tracked individuals and generates InqScribe events for every possible pair. Each individual’s missing‑data periods are also logged as NaN events, helping identify possible interactions or occlusions.

## Requirements

- Python 3.7 or higher
- tkinter (usually included with Python)
- numpy
- matplotlib
- pandas

## Installation

1. Clone the repository or download the source code.
2. Install the required Python packages:
   ```
   pip install numpy matplotlib pandas
   ```
3. Ensure tkinter is installed:
   - On Windows and macOS, it usually comes pre-installed.
   - On Linux, you may need to install it via your package manager, e.g., `sudo apt-get install python3-tk`.

## Running the GUI

Run the main script from the command line:
```
python IDTracker_PostProcessing_GUI.py
```
This will open the GUI window where you can load your data and begin post-processing.

### How the GUI Works

You begin by selecting a **base directory** that contains the outputs from IDTracker.ai. The script automatically scans that directory (and its subfolders) for a `trajectories.csv` file and a `.toml` file that defines the regions of interest (ROIs). If it finds exactly one of each, they will be automatically selected for you. If multiple such files are found, the GUI will show a warning so you can manually pick the correct ones. The script then creates a new folder called `Post_Processing_Output` in that same directory, and all outputs will be saved there by default.

> **Important:** The base directory you select should be the same folder produced by IDTracker.ai for a specific video. This folder must contain both a `trajectories.csv` file (the tracked position data) and a `.toml` file (which defines the arena and bracket ROIs). The script depends on both files to run correctly — if either one is missing, the analysis will not proceed.

If you prefer, you can bypass the base directory scan and manually choose the `trajectories.csv` and `.toml` files yourself using the corresponding browse buttons.

Once your files are selected, you can customize the analysis parameters:
- **Distance threshold (px):** The pixel distance used to define when two beetles are considered in contact.
- **Minimum contact duration (s):** The minimum number of seconds beetles must remain within that distance to count as a proximity event.
- **Output directory:** Optional. If left blank, it will default to the automatically created `Post_Processing_Output` folder.

Press **Run** to execute. When processing completes, a popup will confirm success and list the full paths to all generated files.

### Output Files

The script produces several files summarizing spatial, temporal, and interaction data:

1. **InqScribe‑ready tab‑delimited file** (`*_InqScribe_<threshold>px_<fps>fps.txt`)
   - A file that can be imported directly into InqScribe to visualize and annotate interaction events.
   - Includes both proximity events (where two beetles were within the distance threshold for the required duration) and NaN events (where one beetle’s position is missing—often indicating overlap or fights).

2. **Pairwise Distances CSV** (`*_pairwise_distances_<threshold>px_<fps>fps.csv`)
   - Frame‑by‑frame distances between every possible pair of beetles.
   - Each row corresponds to a single frame and a unique pair of beetles.

3. **Multi‑page Tracks PDF** (`*_tracks_<threshold>px_<fps>fps.pdf`)
   - Each beetle’s trajectory is drawn on its own page, color‑coded consistently across all pages.
   - The final page shows all beetles together.
   - Black polygons show the ROIs (the arena and individual fungus brackets).
   - Triangles mark starting positions; circles mark final positions.

4. **ROI Events CSV** (`*_roi_events.csv`)
   - A log of every time each beetle enters or exits a defined ROI.
   - Useful for quantifying behavioral transitions or detecting when beetles move on/off fungus brackets.

5. **ROI Summary CSV** (`*_roi_summary.csv`)
   - A summary of how much time each beetle spent within each ROI, the distance they traveled inside it, their median speed (calculated over 10‑frame windows) while inside that ROI, and the total number of frames in the video.
   - This file accounts for nested ROIs correctly — for example, the main arena contains all bracket regions, so a beetle inside a bracket is also counted as inside the arena.

All files are saved into the same `Post_Processing_Output` directory unless you choose another location.

---
## Data Dictionary

### 1. InqScribe File (`*_InqScribe_<threshold>px_<fps>fps.txt`)
| Column | Description |
|---------|-------------|
| Start Time | Event start timestamp in HH:MM:SS,mmm format |
| End Time | Event end timestamp in HH:MM:SS,mmm format |
| Title | Text label describing the event (e.g., `Proximity: ID1–ID2`, or `NaN: ID3 missing`) |
| Comment | Detailed information, including frame range and distance/duration notes |

### 2. Pairwise Distances File (`*_pairwise_distances_<threshold>px_<fps>fps.csv`)
| Column | Description |
|---------|-------------|
| frame | Frame number |
| time_s | Time in seconds (relative to start of video) |
| id_i | ID of the first beetle in the pair |
| id_j | ID of the second beetle in the pair |
| distance_px | Distance between the two beetles’ centroids in pixels |
| both_valid | 1 if both beetles were successfully tracked in that frame, else 0 |
| contact_le_<threshold>px | 1 if beetles were within the threshold distance, else 0 |

### 3. ROI Events File (`*_roi_events.csv`)
| Column | Description |
|---------|-------------|
| frame | Frame number of the event |
| time_s | Time of the event in seconds |
| beetle_id | Numeric ID of the beetle involved |
| roi_name | Name of the ROI (e.g., `arena`, `bracket1`, etc.) |
| event_type | `ENTER` or `EXIT` depending on the transition |

### 4. ROI Summary File (`*_roi_summary.csv`)
| Column | Description |
|---------|-------------|
| beetle_id | Numeric ID of the beetle |
| roi_name | ROI label (e.g., `arena`, `bracket2`, etc.) |
| frames_in_roi | Number of frames the beetle was inside the ROI |
| pct_time_in_roi | Fraction of total frames spent in the ROI (0–1) |
| dist_in_roi_px | Total distance traveled (in pixels) while inside the ROI |
| median_speed_10f_px_per_frame | Median speed (in pixels/frame) over 10‑frame rolling windows while the beetle was inside the ROI |
| video_total_frames | Total number of frames in the video (same for all rows; included for clarity) |
| entries_into_roi | The number of times the beetle entered this ROI during the video |

## CLI Fallback

If you prefer, or if the GUI fails to launch, a command-line interface is available for basic correction operations.

## Notes on macOS/tkinter

On macOS, you may need to install Python from python.org to get a working tkinter installation. The system Python often lacks full tkinter support.

## Project Structure

- `IDTracker_PostProcessing_GUI.py` - Main GUI script
- `README.md` - This documentation
- `requirements.txt` - List of required Python packages
- `data/` - Example data files
- `docs/` - Additional documentation and usage examples

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Next Steps
- Run a bunch of tracking runs on differnt cells
- Have it calculate stats
      - Duration in fights (once I have that from step below)
- Have it identify possible fights from log term closeness or closenss followed by NAs
