"""Data loading for the Liu et al. (2020) CME dataset (24-hour horizon).

Sequence construction follows the public CMEpredict.py from
github.com/deepsuncode/RNN-CME-prediction: for every labelled sample, collect it
plus up to 19 earlier samples from the same HARP region (series length 20),
skip rows containing an exact 0.0 (treated as missing), and left-pad with 0.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

ALL_COLS = ["Label", "Timestamp", "NOAA_AR_NUM", "HARP_NUM",
            "TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX",
            "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45", "MEANGAM", "MEANJZH",
            "MEANGBT", "MEANGBZ", "MEANJZD", "MEANGBH", "MEANSHR", "MEANALP"]

# The 12 features used by the lab's LSTM-24 model (CMEpredict.py, lstm / 24 h),
# in the order the script selects them. Same set as Table 1 of
# Rawashdeh, Wang & Herbert (FLAIRS 2025).
FEATURES = ["MEANSHR", "MEANPOT", "SHRGT45", "USFLUX", "MEANJZH", "MEANGAM",
            "SAVNCPP", "ABSNJZH", "TOTPOT", "MEANALP", "TOTUSJZ", "MEANGBZ"]

SERIES_LEN = 20

# The headerless files in the repo's CMEpredict/ folder do NOT store the SHARP
# parameters in the order its ReadMe lists. This is the true order, recovered by
# matching every row (timestamp + region) against the headered files in
# CME_data_samples/ (exact matches, max abs difference 0.0). See
# scripts/check_column_order.py.
CODE_FOLDER_COLS = ["Label", "Timestamp", "NOAA_AR_NUM", "HARP_NUM",
                    "MEANPOT", "SHRGT45", "MEANJZH", "TOTPOT", "MEANSHR", "USFLUX", "MEANALP", "ABSNJZH",
                    "SAVNCPP", "TOTUSJZ", "MEANGBZ", "MEANJZD", "AREA_ACR", "MEANGAM", "TOTUSJH",
                    "R_VALUE", "MEANGBT", "MEANGBH"]


def read(name: str) -> pd.DataFrame:
    """Read one CSV. Headered files use their header; the headerless code-folder file uses the
    recovered true column order (not the ReadMe order)."""
    path = DATA / name
    first = path.open().readline()
    if first.startswith("Label"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None, names=CODE_FOLDER_COLS)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].str.replace("Z", "", regex=False))
    return df


def build_sequences(df: pd.DataFrame, features=FEATURES, series_len=SERIES_LEN):
    """Return X (n, T, F), y (n,), and a metadata frame (one row per sequence)."""
    vals = df[features].to_numpy(dtype=np.float32)
    labels = df["Label"].to_numpy()
    harp = df["HARP_NUM"].to_numpy()
    missing = (vals == 0.0).any(axis=1)
    X, y, keep = [], [], []
    for i in range(len(df)):
        if labels[i] not in ("N", "P") or missing[i]:
            continue
        seq = [vals[i]]
        j = i - 1
        while j >= 0 and len(seq) < series_len and harp[j] == harp[i]:
            seq.insert(0, np.zeros(len(features), np.float32) if missing[j] else vals[j])
            j -= 1
        while len(seq) < series_len:
            seq.insert(0, np.zeros(len(features), np.float32))
        X.append(np.stack(seq))
        y.append(1 if labels[i] == "P" else 0)
        keep.append(i)
    meta = df.iloc[keep][["Label", "Timestamp", "NOAA_AR_NUM", "HARP_NUM"]].reset_index(drop=True)
    return np.stack(X), np.array(y, dtype=np.float32), meta


def load_all(features=FEATURES):
    out = {}
    for key, fname in [("train", "normalized_training_24.csv"),
                       ("test_full", "normalized_testing_24.csv"),
                       ("test_demo", "normalized_testing_24_demo540.csv")]:
        out[key] = build_sequences(read(fname), features)
    return out
