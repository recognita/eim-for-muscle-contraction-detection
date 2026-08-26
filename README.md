# eim-muscle-contraction-detection

Signal processing and machine learning pipeline that classifies movement type from electrical impedance myography (EIM) recordings of opposing forearm muscles, for use in upper-limb prosthesis control.

Published as: Makeeva, D.S., Yakim, M.Y., Kobelev, A.V., Shchukin, S.I. — *Machine Learning Techniques for Muscle Contraction Detection based on Electrical Impedance Measurements*, IEEE USBEREIT 2025, pp. 217–220.

## Overview

Prosthetic hands controlled by surface EMG struggle to disambiguate movement types when muscle signals overlap or are noisy. This project explores electrical impedance myography (EIM) as a complementary signal: measuring the electrical impedance of opposing forearm muscles during different hand/wrist movements, then classifying the movement type from the impedance pattern.

## Method

1. **Filtering** — cleaning raw EIM traces from opposing-muscle channels.
2. **Labelling** — aligning recordings with movement-type labels.
3. **Feature extraction / quantization** — an improved filtering and quantization approach for dataset preparation (the main methodological contribution).
4. **Learning** — classification of movement type from the processed EIM features.

## Results

- Up to **0.95 classification accuracy**, with strong sensitivity across different movement classes.
- Improving the filtering/quantization step was the main driver of the accuracy gain over a naive baseline.
- Noted next step: larger dataset and more complex models to push performance further.

## Repository contents

| File | Purpose |
|---|---|
| `src/filtering.py` | Signal cleaning of raw EIM channels |
| `src/labeling.py` | Movement-type label alignment and windowing |
| `src/dataframe.py` | Recording metadata table (sampling rate, duration, QC) |
| `src/learning.py` | Model training and evaluation *(add your own version here — see note below)* |

> `Learning.ipynb` and `challenge1.ipynb` from the old notebook layout are not included here. `challenge1.ipynb` was an unrelated notebook (a different course assignment, not part of this project) — leave it out. `Learning.ipynb` holds the actual model training/evaluation code and should be added back as `src/learning.py`, cleaned up the same way as the other files here.

## Setup

```bash
pip install -r requirements.txt
python src/filtering.py <input_folder> <output_folder>
python src/dataframe.py <output_folder> <metadata_folder>
python src/labeling.py <file_list.csv> <labeled_output.xlsx>
```

## Tech stack

Python, NumPy, Pandas, scikit-learn, SciPy (signal filtering)
