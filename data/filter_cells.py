import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

#### Cell filtering code. We drop out cells from the dataset which have less than 20 frames and with the time gap between frames bigger than 5


def make_keep_mask(times, min_frames=20, max_step=5):
    mask = []

    for t in times:
        t = np.asarray(t)

        keep = (
            len(t) >= min_frames
            and len(t) > 1
            and np.all(np.diff(t) <= max_step)
        )

        mask.append(bool(keep))

    return np.array(mask, dtype=bool)


def build_mapping(mask):
    rows = []
    new_cell = 1

    for old_cell, keep in enumerate(mask, start=1):
        if keep:
            rows.append((old_cell, new_cell))
            new_cell += 1

    return pd.DataFrame(rows, columns=["old_cell", "new_cell"])


def save_filtered_arrays(times, centroids, riemann, mask, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "times_filtered.npy",
        np.array([x for x, keep in zip(times, mask) if keep], dtype=object),
        allow_pickle=True
    )

    np.save(
        output_dir / "centroids_filtered.npy",
        np.array([x for x, keep in zip(centroids, mask) if keep], dtype=object),
        allow_pickle=True
    )

    np.save(
        output_dir / "riemann_filtered.npy",
        np.array([x for x, keep in zip(riemann, mask) if keep], dtype=object),
        allow_pickle=True
    )


def copy_filtered_cells(cells_dir, output_cells_dir, mapping):
    cells_dir = Path(cells_dir)
    output_cells_dir = Path(output_cells_dir)

    if output_cells_dir.exists():
        shutil.rmtree(output_cells_dir)

    output_cells_dir.mkdir(parents=True, exist_ok=True)

    for _, row in mapping.iterrows():
        old_cell = int(row["old_cell"])
        new_cell = int(row["new_cell"])

        src = cells_dir / f"cell_{old_cell}"
        dst = output_cells_dir / f"cell_{new_cell}"

        if src.exists():
            shutil.copytree(src, dst)


def write_filtered_h5(h5_path, output_h5_path, mapping):
    h5_path = Path(h5_path)
    output_h5_path = Path(output_h5_path)

    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as src_h5, h5py.File(output_h5_path, "w") as out_h5:
        for _, row in mapping.iterrows():
            old_cell = int(row["old_cell"])
            new_cell = int(row["new_cell"])

            old_key = f"track_{old_cell}"
            new_key = f"track_{new_cell}"

            if old_key in src_h5:
                out_h5.create_dataset(new_key, data=src_h5[old_key][:])


def filter_dataset(
    times_path,
    centroids_path,
    riemann_path,
    h5_path,
    cells_dir,
    output_dir="filtered_data",
    min_frames=20,
    max_step=5
):
    output_dir = Path(output_dir)

    times = np.load(times_path, allow_pickle=True)
    centroids = np.load(centroids_path, allow_pickle=True)
    riemann = np.load(riemann_path, allow_pickle=True)

    if not (len(times) == len(centroids) == len(riemann)):
        raise ValueError("times, centroids and riemann arrays must have the same length")

    mask = make_keep_mask(
        times,
        min_frames=min_frames,
        max_step=max_step
    )

    mapping = build_mapping(mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_dir / "old_to_new_mapping.csv", index=False)

    save_filtered_arrays(
        times=times,
        centroids=centroids,
        riemann=riemann,
        mask=mask,
        output_dir=output_dir
    )

    copy_filtered_cells(
        cells_dir=cells_dir,
        output_cells_dir=output_dir / "cells_filtered",
        mapping=mapping
    )

    write_filtered_h5(
        h5_path=h5_path,
        output_h5_path=output_dir / "time_events_filtered.h5",
        mapping=mapping
    )

    return mapping


if __name__ == "__main__":
    mapping = filter_dataset(
        times_path="nov30/times.npy",
        centroids_path="nov30/centroids.npy",
        riemann_path="nov30/riemann_distances.npy",
        h5_path="time_events_95_corrected.h5",
        cells_dir="cells",
        output_dir="filtered_data",
        min_frames=20,
        max_step=5
    )

    print(f"Kept {len(mapping)} tracks.")
