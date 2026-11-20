"""Check the column order of the headerless CSVs in the RNN-CME-prediction repo's CMEpredict/ folder.

The ReadMe in that folder says columns 5-22 hold the 18 SHARP parameters in the order
TOTUSJH, TOTPOT, TOTUSJZ, ..., MEANSHR, MEANALP (the same order as the header row of the
files in CME_data_samples/). This script:

1. matches every labelled row of the CMEpredict/ files to the headered CME_data_samples/
   file (same timestamp and region) and finds, for each column, which named column holds
   identical values;
2. runs a physical sanity check: pairs of SHARP parameters that are known to move together
   (total unsigned flux vs. region area, the two total-current measures, mean shear vs.
   fraction of strong shear) should be strongly correlated under the correct names;
3. computes the six correlations quoted in the text of Rawashdeh, Wang & Herbert (FLAIRS 2025)
   on the 540-sample test file, reading it with the ReadMe order.

Run:  python scripts/check_column_order.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
README_ORDER = ["Label", "Timestamp", "NOAA_AR_NUM", "HARP_NUM",
                "TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX", "AREA_ACR", "MEANPOT",
                "R_VALUE", "SHRGT45", "MEANGAM", "MEANJZH", "MEANGBT", "MEANGBZ", "MEANJZD", "MEANGBH",
                "MEANSHR", "MEANALP"]
SHARP = README_ORDER[4:]


def recover_order(headerless: Path, headered: Path):
    a = pd.read_csv(headerless, header=None, names=README_ORDER)
    a = a[a["Label"].isin(["N", "P"])]
    b = pd.read_csv(headered)
    m = a.merge(b, on=["Timestamp", "NOAA_AR_NUM"], suffixes=("_pos", "_true"))
    mapping = {}
    for pos_name in SHARP:
        errs = {t: float(np.abs(m[pos_name + "_pos"] - m[t + "_true"]).max()) for t in SHARP}
        best = min(errs, key=errs.get)
        mapping[pos_name] = (best, errs[best])
    return len(m), mapping


def main():
    n, mapping = recover_order(DATA / "codefolder_normalized_training_24.csv", DATA / "normalized_training_24.csv")
    print(f"Matched {n} labelled training rows.\n")
    print(f"{'ReadMe name (by position)':28s} -> {'column that actually holds those values':40s} max|diff|")
    for k, (v, e) in mapping.items():
        flag = "" if k == v else "   <- differs"
        print(f"{k:28s} -> {v:40s} {e:.1e}{flag}")

    print("\nPhysical sanity check (Pearson r on labelled training rows):")
    pairs = [("USFLUX", "AREA_ACR"), ("TOTUSJH", "TOTUSJZ"), ("MEANSHR", "SHRGT45"), ("MEANJZH", "MEANALP")]
    headered = pd.read_csv(DATA / "normalized_training_24.csv")
    pos = pd.read_csv(DATA / "codefolder_normalized_training_24.csv", header=None, names=README_ORDER)
    pos = pos[pos["Label"].isin(["N", "P"])]
    for a, b in pairs:
        print(f"  {a:8s} vs {b:8s}: headered file r = {headered[a].corr(headered[b]):+.2f} | "
              f"code-folder file read in ReadMe order r = {pos[a].corr(pos[b]):+.2f}")

    print("\nCorrelations quoted in FLAIRS 2025 vs. the 540-sample test file read in ReadMe order:")
    demo = pd.read_csv(DATA / "normalized_testing_24_demo540.csv", header=None, names=README_ORDER)
    demo = demo[demo["Label"].isin(["N", "P"])]
    quoted = [("TOTPOT", "SAVNCPP", 1.0), ("TOTPOT", "TOTUSJZ", 0.05), ("TOTPOT", "MEANGAM", -0.69),
              ("MEANALP", "MEANSHR", 0.96), ("MEANALP", "MEANGBZ", -0.09), ("MEANALP", "USFLUX", -0.97)]
    for a, b, q in quoted:
        r = demo[a].corr(demo[b])
        print(f"  {a:7s}-{b:8s} paper {q:+.2f} | file (ReadMe order) {r:+.2f} | "
              f"actual columns: {mapping[a][0]}-{mapping[b][0]}")


if __name__ == "__main__":
    main()
