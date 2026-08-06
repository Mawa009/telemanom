"""
Converts a folder of CSVs (same sensor columns, different time chunks) into
per-sensor .npy files expected by telemanom's channel.py, while recording
file-boundary indices so windows never cross between two source files.

Every column must be numeric (header row = sensor/column names).
Files are concatenated in TIME ORDER based on filename sort order, so
name your files so they sort correctly (e.g. chunk_01.csv, chunk_02.csv,
or 2024-01-01.csv, 2024-01-02.csv).
"""
import numpy as np
import pandas as pd
import os
import glob


def convert_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {input_folder}")

    # files are already sorted by filename above -> treated as time order
    dfs = [(f, pd.read_csv(f)) for f in csv_files]

    sensor_cols = dfs[0][1].columns  # ALL columns are sensors now (no timestamp)

    # concatenate each sensor's values across files, track boundaries
    combined = {col: [] for col in sensor_cols}
    boundaries = []
    running_total = 0

    for fname, df in dfs:
        for col in sensor_cols:
            combined[col].append(df[col].values)
        running_total += len(df)
        boundaries.append(running_total)  # end index of this file (exclusive)

    boundaries = np.array(boundaries[:-1])  # drop last (end of full array, not a real cut)

    for col in sensor_cols:
        arr = np.concatenate(combined[col]).reshape(-1, 1)
        np.save(os.path.join(output_folder, f"{col}.npy"), arr)

    np.save(os.path.join(output_folder, "_boundaries.npy"), boundaries)

    print(f"Converted {len(csv_files)} files -> {len(sensor_cols)} channels")
    print(f"Sensors: {list(sensor_cols)}")
    print(f"Boundary indices (rows where a new file starts): {boundaries}")
    return sensor_cols, boundaries


if __name__ == "__main__":
    convert_folder("train", "data/train")
    convert_folder("test", "data/test")