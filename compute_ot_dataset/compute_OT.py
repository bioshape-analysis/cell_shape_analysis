from pathlib import Path

import numpy as np
import ot
from scipy.spatial import distance_matrix


def compute_wasserstein_distance(outline_1, outline_2):
    """
    Natalia-style W2 distance:
        W2 = sqrt(emd2(a, b, ||x-y||^2))
    """
    n1 = outline_1.shape[0]
    n2 = outline_2.shape[0]

    weights_1 = np.ones(n1) / n1
    weights_2 = np.ones(n2) / n2

    D12 = distance_matrix(outline_1, outline_2, p=2)
    cost_matrix = np.square(D12)

    return np.sqrt(
        ot.emd2(weights_1, weights_2, cost_matrix)
    )


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


def center_outline(outline, centroid):
    return np.asarray(outline, dtype=float) - np.asarray(centroid, dtype=float)


def load_cell_data_sorted_by_time(cell_dir):
    frame_dirs = get_frame_dirs(cell_dir)

    if len(frame_dirs) == 0:
        raise ValueError(f"No frame directories found in {cell_dir}")

    records = []

    for frame_dir in frame_dirs:
        time = load_time(frame_dir)
        centroid = load_centroid(frame_dir)
        outline = load_outline(frame_dir)

        centered_outline = center_outline(outline, centroid)

        records.append(
            {
                "time": time,
                "centroid": centroid,
                "outline": centered_outline,
            }
        )

    records = sorted(records, key=lambda x: x["time"])

    times = np.asarray([r["time"] for r in records], dtype=int)
    centroids = np.asarray([r["centroid"] for r in records], dtype=float)
    outlines = [r["outline"] for r in records]

    return outlines, times, centroids


def compute_cell_distances(cell_dir):
    outlines, times, centroids = load_cell_data_sorted_by_time(cell_dir)

    n_frames = len(outlines)

    distances = np.zeros(n_frames, dtype=float)

    for frame_idx in range(1, n_frames):
        distances[frame_idx] = compute_wasserstein_distance(
            outlines[frame_idx - 1],
            outlines[frame_idx]
        )

    first_outline = outlines[0]
    last_outline = outlines[-1]

    first_last_distance = compute_wasserstein_distance(
        first_outline,
        last_outline
    )

    return (
        distances,
        times,
        centroids,
        first_outline,
        last_outline,
        first_last_distance
    )


def compute_dataset_distances(
    cells_dir,
    output_dir,
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

    first_shapes = []
    last_shapes = []
    first_last_shape_distances = []

    for cell_idx, cell_dir in enumerate(cell_dirs, start=1):
        if verbose:
            print(f"Processing {cell_dir.name} ({cell_idx}/{len(cell_dirs)})")

        (
            distances,
            times,
            centroids,
            first_outline,
            last_outline,
            first_last_distance
        ) = compute_cell_distances(cell_dir=cell_dir)

        all_distances.append(distances)
        all_times.append(times)
        all_centroids.append(centroids)

        first_shapes.append(first_outline)
        last_shapes.append(last_outline)
        first_last_shape_distances.append(first_last_distance)

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

    np.save(
        output_dir / "first_shapes.npy",
        np.array(first_shapes, dtype=object),
        allow_pickle=True
    )

    np.save(
        output_dir / "last_shapes.npy",
        np.array(last_shapes, dtype=object),
        allow_pickle=True
    )

    np.save(
        output_dir / "first_last_shape_distances.npy",
        np.array(first_last_shape_distances),
        allow_pickle=True
    )

    return (
        all_distances,
        all_times,
        all_centroids,
        first_shapes,
        last_shapes,
        first_last_shape_distances
    )


if __name__ == "__main__":
    compute_dataset_distances(
        cells_dir="../data/cells_filtered",
        output_dir="",
        verbose=True
    )
