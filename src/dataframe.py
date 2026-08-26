"""
Builds a metadata DataFrame (sampling rate, duration, whether the recording
matches the expected 125 Hz sampling rate, and the per-recording median
impedance-change coefficient) over a folder of filtered EIM recordings.

Cleaned up from Dataframe.ipynb, with one real bug fixed: the original
`except` branch built a 5-element row (`[file_path, file_name, 'Error',
'Error', False]`) for a 7-column DataFrame (`file_path, name, creation
date, sampling rate, signal_duration, appropriate, mediana`). Assigning a
5-element list to a 7-column `df.loc[index]` raises
`ValueError: cannot set a row with mismatched columns` in pandas — so the
very first corrupted/unreadable file in a batch would crash the whole run
instead of being logged and skipped. Fixed by padding the error row to
match every column.
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

EXPECTED_SAMPLING_RATE_HZ = 125
SAMPLING_RATE_TOLERANCE_HZ = 1
COLUMNS = ["file_path", "name", "creation date", "sampling rate", "signal_duration", "appropriate", "mediana"]


def create_data_frame(file_paths, save_folder, save_name="recordings_metadata.csv"):
    df = pd.DataFrame(columns=COLUMNS)

    for index, file_path in tqdm(enumerate(file_paths), total=len(file_paths)):
        file_name = os.path.basename(file_path)
        try:
            csv_data = pd.read_excel(file_path)
            creation_date = file_name[:-7]
            csv_data["Time"] = pd.to_numeric(csv_data["Time"], errors="coerce")
            time = csv_data["Time"]

            median_value = csv_data["median_summ_abs_coef"].iloc[0]
            sampling_rate = 1 / (time[3] - time[4])
            signal_duration = len(csv_data) / sampling_rate
            appropriate = abs(sampling_rate - EXPECTED_SAMPLING_RATE_HZ) < SAMPLING_RATE_TOLERANCE_HZ

            df.loc[index] = [file_path, file_name, creation_date, sampling_rate, signal_duration, appropriate, median_value]
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            # Row must have one value per column in COLUMNS, or pandas raises
            # ValueError and aborts the whole batch (this was the bug).
            df.loc[index] = [file_path, file_name, "Error", None, None, False, None]

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, save_name)
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"DataFrame created and saved to {save_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder_path", help="Folder of filtered .xlsx EIM recordings")
    parser.add_argument("save_folder", help="Folder to write the metadata CSV to")
    args = parser.parse_args()

    paths = [os.path.join(args.folder_path, f) for f in sorted(os.listdir(args.folder_path)) if f.endswith(".xlsx")]
    create_data_frame(paths, args.save_folder)
