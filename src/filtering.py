"""
Signal cleaning for raw two-channel EIM (electrical impedance myography)
recordings: exponential smoothing, Hilbert-transform envelope/phase
extraction, and phase-sign classification into "grasp" / "rotation"
movement events.

Cleaned up from Filtering.ipynb: translated print/error strings to
English, made plotting opt-in (the original always opened 4 matplotlib
figures per file — fine for one file in a notebook, but calling this in
a loop over a whole folder silently accumulates dozens of open figures
and eats memory; `plot=False` by default, with `plt.close("all")`
afterwards when it's off), and wrapped the logic behind a CLI so paths
aren't hardcoded. The signal-processing logic itself (the 0.03125
exponential-smoothing coefficient, the 61-tap FIR differentiator, the
0.15 / 0.05 noise-threshold split, the ±0.75 vs ±2.35 phase thresholds)
is unchanged from the original — those are your calibrated parameters,
not something to "clean up" without re-validating against data.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert

# Exponential-smoothing coefficient applied to each impedance channel.
SMOOTHING_ALPHA = 0.03125
# Sliding window (samples) used for the Hilbert-transform envelope/phase.
ENVELOPE_WINDOW = 80
# Length of the FIR differentiator applied to the instantaneous phase.
DIFFERENTIATOR_LENGTH = 61
# Phase-threshold pair used depending on how noisy the recording is.
NOISY_THRESHOLD_RANGE = (0.05, 0.15)
PHASE_THRESHOLD_LOW_NOISE = 0.75
PHASE_THRESHOLD_HIGH_NOISE = 2.35


def _exponential_smooth(values, alpha):
    smoothed = [values.iloc[0] * alpha if len(values) and values.iloc[0] is not None else 0]
    for i in range(1, len(values)):
        smoothed.append(alpha * values.iloc[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed


def _hilbert_envelope_and_phase(signal, window_size=ENVELOPE_WINDOW, step_size=1):
    envelope = np.zeros_like(signal)
    phase = np.zeros_like(signal)
    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i : i + window_size]
        analytic = hilbert(window)
        envelope[i : i + window_size] = np.abs(analytic)
        phase[i : i + window_size] = np.angle(analytic)
    return envelope, phase


def _differentiator_kernel(length=DIFFERENTIATOR_LENGTH):
    """FIR kernel: coeff[k] = -(12k - 6(n-1)) / (n^3 - n), a discrete derivative-like filter."""
    n = length
    return [-(12.0 * k - 6 * (n - 1)) / (n**3 - n) for k in range(n)]


def _signum(x):
    return int(np.copysign(1, x))


def process_file(file_path, save_folder, plot=False):
    """
    Process a single Excel recording: smooth both channels, extract the
    instantaneous phase via a windowed Hilbert transform, classify each
    sample into grasp/rotation events from the phase sign, and write an
    annotated copy of the file to `save_folder`.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_excel(file_path)
    filtered_data = data[data["Time"] >= 0].copy()
    channel1, channel2 = filtered_data.iloc[:, 1], filtered_data.iloc[:, 2]

    if plot:
        plt.figure(figsize=(34, 6))
        plt.rc("font", size=40)
        plt.plot(channel1, linewidth=4)
        plt.title("Channel 1")
        plt.ylabel("R, Ω")

        plt.figure(figsize=(34, 6))
        plt.rc("font", size=40)
        plt.plot(channel2, linewidth=4)
        plt.title("Channel 2")
        plt.ylabel("R, Ω")

    smoothed1 = _exponential_smooth(channel1, SMOOTHING_ALPHA)
    smoothed2 = _exponential_smooth(channel2, SMOOTHING_ALPHA)

    _, phase1 = _hilbert_envelope_and_phase(np.asarray(smoothed1))
    _, phase2 = _hilbert_envelope_and_phase(np.asarray(smoothed2))

    kernel = _differentiator_kernel()
    filtered1 = np.convolve(phase1, kernel, mode="same")
    filtered2 = np.convolve(phase2, kernel, mode="same")
    abs_filtered1, abs_filtered2 = np.abs(filtered1), np.abs(filtered2)
    noise_level = np.median(abs_filtered1 + abs_filtered2)

    low, high = NOISY_THRESHOLD_RANGE
    phase_threshold = PHASE_THRESHOLD_LOW_NOISE if low < noise_level < high else PHASE_THRESHOLD_HIGH_NOISE

    rect1 = [0 if abs(x) < phase_threshold else _signum(x) for x in phase1]
    rect2 = [0 if abs(x) < phase_threshold else _signum(x) for x in phase2]

    grasp = np.zeros_like(rect1)
    rotation = np.zeros_like(rect1)
    for i in range(len(rect1)):
        if rect1[i] == 1 and rect2[i] == 1:
            rotation[i] = 1
        elif rect1[i] == -1 and rect2[i] == -1:
            rotation[i] = -1
        elif rect1[i] == 1 and rect2[i] == -1:
            grasp[i] = 1
        elif rect1[i] == -1 and rect2[i] == 1:
            grasp[i] = -1

    filtered_data["grasp"] = grasp
    filtered_data["rotation"] = rotation
    filtered_data["rect1"] = rect1
    filtered_data["rect2"] = rect2
    filtered_data["coef1"] = filtered1
    filtered_data["coef2"] = filtered2
    filtered_data["abs_coef1"] = abs_filtered1
    filtered_data["abs_coef2"] = abs_filtered2
    filtered_data["summ_abs_coef"] = abs_filtered1 + abs_filtered2
    filtered_data["median_summ_abs_coef"] = None
    filtered_data.loc[filtered_data.index[0], "median_summ_abs_coef"] = filtered_data["summ_abs_coef"].median()

    os.makedirs(save_folder, exist_ok=True)
    output_path = os.path.join(save_folder, os.path.basename(file_path))
    filtered_data.to_excel(output_path, index=False)
    print(f"Processed and saved: {output_path}")

    if plot:
        plt.figure(figsize=(34, 6))
        plt.rc("font", size=40)
        plt.plot(grasp, linewidth=4)
        plt.title(f"Grasp {output_path}")
        plt.xlim(200, 3000)

        plt.figure(figsize=(34, 6))
        plt.rc("font", size=40)
        plt.plot(rotation, linewidth=4)
        plt.title(f"Rotation {output_path}")
        plt.xlim(200, 3000)
    else:
        plt.close("all")

    return filtered_data


def process_all_files(input_folder, save_folder, plot=False):
    os.makedirs(save_folder, exist_ok=True)
    for file_name in sorted(os.listdir(input_folder)):
        if file_name.endswith(".xlsx"):
            process_file(os.path.join(input_folder, file_name), save_folder, plot=plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", help="Folder of raw .xlsx EIM recordings")
    parser.add_argument("save_folder", help="Folder to write annotated recordings to")
    parser.add_argument("--plot", action="store_true", help="Show per-file diagnostic plots (off by default)")
    args = parser.parse_args()

    process_all_files(args.input_folder, args.save_folder, plot=args.plot)
