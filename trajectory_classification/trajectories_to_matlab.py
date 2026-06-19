from pathlib import Path

import numpy as np
from scipy.io import savemat


def build_utrack_format(
    root_dir="filtered_data/cells_filtered",
    save_path="utrack_output.mat"
):
    root_dir = Path(root_dir)
    save_path = Path(save_path)

    cell_dirs = sorted(
        [
            p for p in root_dir.iterdir()
            if p.is_dir() and p.name.startswith("cell_")
        ],
        key=lambda p: int(p.name.split("_")[1])
    )

    if len(cell_dirs) == 0:
        raise ValueError(f"No cell directories found in {root_dir}")

    max_time = 0

    for cell_dir in cell_dirs:
        frame_dirs = sorted(
            [p for p in cell_dir.iterdir() if p.is_dir()],
            key=lambda p: int(p.name.split("_")[1]) if "_" in p.name else 0
        )

        for frame_dir in frame_dirs:
            time_path = frame_dir / "time.npy"

            if time_path.exists():
                t = int(np.load(time_path).item())
                max_time = max(max_time, t)

    if max_time == 0:
        raise ValueError("No valid time.npy files found.")

    tracks_matrix = np.full(
        (len(cell_dirs), 8 * max_time),
        np.nan
    )

    for cell_idx, cell_dir in enumerate(cell_dirs):
        frame_dirs = sorted(
            [p for p in cell_dir.iterdir() if p.is_dir()],
            key=lambda p: int(p.name.split("_")[1]) if "_" in p.name else 0
        )

        for frame_dir in frame_dirs:
            time_path = frame_dir / "time.npy"
            centroid_path = frame_dir / "centroid.npy"

            if not time_path.exists() or not centroid_path.exists():
                continue

            t = int(np.load(time_path).item())
            centroid = np.load(centroid_path)

            base_col = 8 * (t - 1)

            tracks_matrix[cell_idx, base_col] = centroid[0]
            tracks_matrix[cell_idx, base_col + 1] = centroid[1]
            tracks_matrix[cell_idx, base_col + 2:base_col + 8] = 0.0

    tracks_struct = {
        f"track_{i + 1}": tracks_matrix[i, :]
        for i in range(len(cell_dirs))
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(save_path, {"tracks": tracks_struct})

    return tracks_matrix


if __name__ == "__main__":
    build_utrack_format()
