"""
Windowing and labeling of filtered EIM recordings (output of `filtering.py`)
into fixed-length segments with a grasp/rotation label per segment, plus an
FFT-magnitude feature per channel.

Cleaned up from Labeling.ipynb: added docstrings/type hints, translated
comments, and replaced the hardcoded `"Your_DataFrame.csv"` /
`"Your_Final_Data_Frame"` filenames with CLI arguments. The windowing and
labeling logic (12-sample windows, majority-sign labeling) is unchanged.
"""

import argparse
import os

import numpy as np
import pandas as pd

WINDOW_SIZE = 12


def classify_grasp(grasp_values):
    """Label a window as a positive/negative grasp event, or none, from the signed grasp series."""
    total, magnitude = np.sum(grasp_values), np.sum(np.abs(grasp_values))
    if magnitude == 0:
        return "No movement"
    if total == magnitude:
        return "Positive movement"
    if total == -magnitude:
        return "Negative movement"
    return None


def classify_rotation(rotation_values):
    """Label a window as a positive/negative rotation event, or none, from the signed rotation series."""
    total, magnitude = np.sum(rotation_values), np.sum(np.abs(rotation_values))
    if magnitude == 0:
        return "No movement"
    if total == magnitude:
        return "Positive movement"
    if total == -magnitude:
        return "Negative movement"
    return None


def process_data(df, window_size=WINDOW_SIZE):
    """Group a filtered recording into fixed-size windows and label each one."""
    df = df.copy()
    df["window"] = (df.index // window_size).astype(int)

    grouped = df.groupby("window").agg({
        "Z1": list, "Z2": list, "grasp": list, "rotation": list,
        "abs_coef1": list, "abs_coef2": list,
    }).reset_index()

    grouped["Z1_fft"] = grouped["Z1"].apply(lambda x: list(np.abs(np.fft.fft(x))))
    grouped["Z2_fft"] = grouped["Z2"].apply(lambda x: list(np.abs(np.fft.fft(x))))
    grouped["label_grasp"] = grouped["grasp"].apply(classify_grasp)
    grouped["label_rotation"] = grouped["rotation"].apply(classify_rotation)

    return grouped.drop(columns=["grasp", "rotation", "abs_coef1", "abs_coef2"])


def build_labeled_dataset(file_list_csv):
    """Run `process_data` over every recording listed in `file_list_csv` (must have a `file_path` column)."""
    file_list = pd.read_csv(file_list_csv, encoding="utf-8")

    all_features = []
    for _, row in file_list.iterrows():
        file_path = row["file_path"]
        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue
        df = pd.read_excel(file_path)
        processed = process_data(df)
        processed["file_name"] = os.path.basename(file_path)
        all_features.append(processed)

    if not all_features:
        raise RuntimeError("No input files were found — check the paths in the file list CSV.")

    return pd.concat(all_features, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_list_csv", help="CSV with a 'file_path' column listing filtered recordings")
    parser.add_argument("output_path", help="Where to write the labeled feature table (.xlsx)")
    args = parser.parse_args()

    final_df = build_labeled_dataset(args.file_list_csv)
    print(final_df.head())
    final_df.to_excel(args.output_path, index=False)
    print(f"Labeled dataset written to {args.output_path}")
