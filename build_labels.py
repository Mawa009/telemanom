"""
Converts a folder of binary anomaly-mask CSVs into the labeled_anomalies.csv
format telemanom's Detector expects.

Each file in the labels folder must have the SAME FILENAME as its matching
file in the test/ folder (e.g. labels/chunk_00.csv <-> test/chunk_00.csv),
and contain a single column of 0/1 values with the SAME NUMBER OF ROWS as
that test file. 1 = anomaly, 0 = normal.

Since one mask applies to the whole timestep (not a specific sensor), the
resulting anomaly windows are applied identically to every sensor channel.

Usage:
    python build_labels.py test/ labels/ labeled_anomalies.csv
"""
import pandas as pd
import numpy as np
import os
import glob
import sys


def mask_to_sequences(mask):
    """Convert a 0/1 array into a list of [start, end] inclusive index pairs."""
    sequences = []
    in_run = False
    start = None
    for i, v in enumerate(mask):
        if v == 1 and not in_run:
            start = i
            in_run = True
        elif v == 0 and in_run:
            sequences.append([start, i - 1])
            in_run = False
    if in_run:
        sequences.append([start, len(mask) - 1])
    return sequences


def build_labels(test_folder, labels_folder, output_csv):
    test_files = sorted(glob.glob(os.path.join(test_folder, "*.csv")))
    if not test_files:
        raise FileNotFoundError(f"No CSVs found in {test_folder}")

    # sensor names come from the test files' columns
    sensor_cols = list(pd.read_csv(test_files[0]).columns)

    full_mask = []
    for tf in test_files:
        base = os.path.basename(tf)
        label_file = os.path.join(labels_folder, base)
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Missing matching label file: {label_file}")

        test_df = pd.read_csv(tf)
        label_df = pd.read_csv(label_file)

        if len(label_df) != len(test_df):
            raise ValueError(
                f"{label_file} has {len(label_df)} rows but {tf} has {len(test_df)} rows"
            )

        mask_col = label_df.columns[0]
        full_mask.append(label_df[mask_col].values)

    full_mask = np.concatenate(full_mask)
    sequences = mask_to_sequences(full_mask)

    rows = []
    for sensor in sensor_cols:
        rows.append({
            "chan_id": sensor,
            "spacecraft": "custom",
            "anomaly_sequences": str(sequences),
            "class": "custom",
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Total test rows: {len(full_mask)}")
    print(f"Anomaly windows found: {sequences}")
    print(f"Wrote {len(rows)} channel(s) to {output_csv}")
    return out_df


if __name__ == "__main__":
    test_folder = sys.argv[1] if len(sys.argv) > 1 else "test"
    labels_folder = sys.argv[2] if len(sys.argv) > 2 else "labels"
    output_csv = sys.argv[3] if len(sys.argv) > 3 else "labeled_anomalies.csv"
    build_labels(test_folder, labels_folder, output_csv)