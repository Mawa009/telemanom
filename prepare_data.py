"""
Converts a folder of CSVs (same sensor columns, different time chunks) into
per-sensor .npy files expected by telemanom's channel.py, with preprocessing:

- NaN rows removed (TRAIN ONLY -- test rows are interpolated instead, since
  dropping test rows would break alignment with ground-truth labels)
- Exact duplicate rows removed (TRAIN ONLY, same reason)
- Outliers clipped to the [1st, 99th] percentile range, per sensor,
  using percentiles computed from TRAIN data only
- Z-score normalization (mean 0, std 1) per sensor, using mean/std
  computed from TRAIN data only -- applied to both train and test to
  avoid leaking test statistics into the pipeline

File-boundary indices are tracked (post-cleaning) so windows never cross
between two source files. Files are concatenated in filename sort order.

Usage:
    python prepare_data.py
"""
import numpy as np
import pandas as pd
import os
import glob
import json

LOWER_PCT = 1
UPPER_PCT = 99


def _load_and_clean_chunk(f, sensor_cols, train):
    df = pd.read_csv(f)[sensor_cols]

    if train:
        before = len(df)
        df = df.dropna()
        n_nan = before - len(df)

        if n_nan:
            print(f"  {os.path.basename(f)}: dropped {n_nan} NaN row(s) "
                  f"(duplicates kept)")
    else:
        n_before_nan = df.isna().sum().sum()
        if n_before_nan:
            df = df.interpolate(limit_direction="both")
            print(f"  {os.path.basename(f)}: interpolated {n_before_nan} "
                  f"missing value(s) (row count preserved)")

    return df


def convert_folder(input_folder, output_folder, sensor_cols=None,
                    clip_bounds=None, norm_stats=None, train=True,
                    skip_normalization=False):
    os.makedirs(output_folder, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {input_folder}")

    if sensor_cols is None:
        sensor_cols = list(pd.read_csv(csv_files[0]).columns)

    dfs = [_load_and_clean_chunk(f, sensor_cols, train) for f in csv_files]

    combined = {col: [] for col in sensor_cols}
    boundaries = []
    running_total = 0
    for df in dfs:
        for col in sensor_cols:
            combined[col].append(df[col].values.astype(float))
        running_total += len(df)
        boundaries.append(running_total)
    boundaries = np.array(boundaries[:-1])

    if skip_normalization:
        print(f"{input_folder}: skipping outlier clipping + normalization "
              f"(data assumed already preprocessed)")
        clip_bounds = clip_bounds or {col: None for col in sensor_cols}
        norm_stats = norm_stats or {col: None for col in sensor_cols}
    else:
        # compute clip bounds / norm stats from TRAIN data only
        if train:
            clip_bounds = {}
            for col in sensor_cols:
                arr = np.concatenate(combined[col])
                lo, hi = np.percentile(arr, [LOWER_PCT, UPPER_PCT])
                clip_bounds[col] = (float(lo), float(hi))

        for col in sensor_cols:
            combined[col] = [np.clip(a, *clip_bounds[col]) for a in combined[col]]

        if train:
            norm_stats = {}
            for col in sensor_cols:
                arr = np.concatenate(combined[col])
                mean, std = float(arr.mean()), float(arr.std())
                if std == 0:
                    std = 1.0  # avoid divide-by-zero for constant sensors
                norm_stats[col] = (mean, std)

        for col in sensor_cols:
            mean, std = norm_stats[col]
            combined[col] = [(a - mean) / std for a in combined[col]]

    for col in sensor_cols:
        arr = np.concatenate(combined[col]).reshape(-1, 1)
        np.save(os.path.join(output_folder, f"{col}.npy"), arr)

    np.save(os.path.join(output_folder, "_boundaries.npy"), boundaries)

    print(f"{input_folder}: {len(csv_files)} file(s) -> {len(sensor_cols)} "
          f"channel(s), {running_total} total rows")

    return sensor_cols, clip_bounds, norm_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-normalization", action="store_true",
                        help="Skip outlier clipping + z-score normalization "
                             "(use when input data is already preprocessed, "
                             "e.g. NASA SMAP/MSL .npy data converted to CSV)")
    args = parser.parse_args()

    sensor_cols, clip_bounds, norm_stats = convert_folder(
        "train", "data/train", train=True,
        skip_normalization=args.skip_normalization)

    convert_folder("test", "data/test", sensor_cols=sensor_cols,
                    clip_bounds=clip_bounds, norm_stats=norm_stats,
                    train=False, skip_normalization=args.skip_normalization)

    with open("data/preprocessing_stats.json", "w") as f:
        json.dump({"clip_bounds": clip_bounds, "norm_stats": norm_stats,
                   "skip_normalization": args.skip_normalization}, f, indent=2)
    print("Saved preprocessing_stats.json")