from pathlib import Path

import numpy as np
import ot
from scipy.spatial import distance_matrix

from src.interpolation import interpolate, preprocess
from src.alignment import align


def compute_wasserstein_distance(outline_1, outline_2):
    n1 = outline_1.shape[0]
    n2 = outline_2.shape[0]

    weights_1 = np.ones(n1) / n1
    weights_2 = np.ones(n2) / n2

    cost_matrix = distance_matrix(outline_1, outline_2, p=2)

    return ot.emd2(weights_1, weights_2, cost_matrix)


def get_cell_dirs(cells_dir):
    cells_dir = Path(cells_dir)

    return sorted(
        [
            p for p in cells_dir.iterdir()
            if p.is_dir() and p.name.startswith("cell_")
        ],
        key=lambda p: int(p.name.split("_")[1])
    )


def get_frame_dirs(cell_dir):
    return sorted(
        [
            p for p in cell_dir.iterdir()
            if p.is_dir() and p.name.startswith("frame_")
        ],
        key=lambda p: int(p.name.split("_")[1])
    )


def load_outline(frame_dir):
    return np.load(frame_dir / "outline.npy")


def load_time(frame_dir):
    return int(np.load(frame_dir / "time.npy").item())


def load_centroid(frame_dir):
    return np.load(frame_dir / "centroid.npy")


def preprocess_outline(outline, n_points=200):
    outline = interpolate(outline, n_points)
    outline = preprocess(outline)

    return outline


def compute_cell_distances(cell_dir, n_points=200):
    frame_dirs = get_frame_dirs(cell_dir)

    if len(frame_dirs) == 0:
        raise ValueError(f"No frame directories found in {cell_dir}")

    n_frames = len(frame_dirs)

    distances = np.zeros(n_frames)
    times = np.zeros(n_frames)
    centroids = np.zeros((n_frames, 2))

    reference_outline = preprocess_outline(
        load_outline(frame_dirs[0]),
        n_points=n_points
    )

    reference_outline = align(
        reference_outline,
        reference_outline,
        rescale=True,
        rotation=False,
        reparameterization=True,
        k_sampling_points=n_points
    )

    for frame_idx, frame_dir in enumerate(frame_dirs):
        outline = preprocess_outline(
            load_outline(frame_dir),
            n_points=n_points
        )

        aligned_outline = align(
            outline,
            reference_outline,
            rescale=False,
            rotation=False,
            reparameterization=False,
            k_sampling_points=n_points
        )

        distances[frame_idx] = compute_wasserstein_distance(
            aligned_outline,
            reference_outline
        )

        times[frame_idx] = load_time(frame_dir)
        centroids[frame_idx] = load_centroid(frame_dir)

        reference_outline = aligned_outline

    return distances, times, centroids


def compute_dataset_distances(
    cells_dir,
    output_dir,
    n_points=200,
    verbose=False
):
    cells_dir = Path(cells_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_dirs = get_cell_dirs(cells_dir)

    if len(cell_dirs) == 0:
        raise ValueError(f"No cell directories found in {cells_dir}")

    all_distances = []
    all_times = []
    all_centroids = []

    for cell_idx, cell_dir in enumerate(cell_dirs, start=1):
        if verbose:
            print(f"Processing {cell_dir.name} ({cell_idx}/{len(cell_dirs)})")

        distances, times, centroids = compute_cell_distances(
            cell_dir=cell_dir,
            n_points=n_points
        )

        all_distances.append(distances)
        all_times.append(times)
        all_centroids.append(centroids)

    np.save(
        output_dir / "ot_distances.npy",
        np.array(all_distances, dtype=object),
        allow_pickle=True
    )

    np.save(
        output_dir / "times.npy",
        np.array(all_times, dtype=object),
        allow_pickle=True
    )

    np.save(
        output_dir / "centroids.npy",
        np.array(all_centroids, dtype=object),
        allow_pickle=True
    )

    return all_distances, all_times, all_centroids


if __name__ == "__main__":
    compute_dataset_distances(
        cells_dir=".",
        output_dir=".",
        n_points=200,
        verbose=True
    )
