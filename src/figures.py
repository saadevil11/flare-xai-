"""Make the four figures used in the README."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_tss():
    m = pd.read_csv(RES / "metrics_lstm.csv")
    b = pd.read_csv(RES / "metrics_baselines.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    rows = [("test_demo", "Paper's test set\n540 samples, 2 regions"),
            ("test_full", "All 2016-17 test data\n1,981 samples, 7 regions")]
    rng = np.random.default_rng(0)
    for yi, (key, lab) in enumerate(rows):
        v = m[m.set == key]
        jitter = rng.uniform(-0.08, 0.08, len(v))
        ax.scatter(v.TSS, yi + jitter, s=46, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3,
                   label="LSTM-attention, 10 seeds" if yi == 0 else None)
        lr = b[(b.model == "logreg_last_step") & (b.set == key)].TSS.iloc[0]
        ax.scatter([lr], [yi], marker="D", s=46, facecolor=SURF, edgecolor=ORANGE, linewidth=2, zorder=4,
                   label="Logistic regression baseline" if yi == 0 else None)
        if key == "test_demo":
            lab_tss = b[b.model == "lab_saved_lstm24"].TSS.iloc[0]
            ax.scatter([lab_tss], [yi - 0.25], marker="v", s=46, color=INK2, zorder=4, label="Lab's saved model")
        ax.text(1.0, yi + 0.3, f"mean {v.TSS.mean():.2f} (range {v.TSS.min():.2f}-{v.TSS.max():.2f})", va="center", ha="right", color=INK, fontsize=8.5)
    ax.set_yticks([0, 1], [r[1] for r in rows]); ax.set_ylim(-0.6, 1.5)
    ax.set_xlim(0, 1.04); ax.set_xlabel("True Skill Statistic (TSS), 24-hour flare-CME association")
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.28), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig1_tss_two_test_sets.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_rank_heatmap():
    r = pd.read_csv(RES / "shap_rank_test_full.csv", index_col=0)
    r = r.loc[r.median(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ[::-1])
    ax.imshow(r.values, cmap=cmap, vmin=1, vmax=12, aspect="auto")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            val = r.values[i, j]
            ax.text(j, i, str(val), ha="center", va="center", fontsize=8.5, color=SURF if val <= 6 else INK)
    ax.set_xticks(range(r.shape[1]), [c.replace("seed", "") for c in r.columns])
    ax.set_yticks(range(r.shape[0]), [f"{f}  ({r.loc[f].min()}-{r.loc[f].max()})" for f in r.index])
    ax.set_xlabel("Training seed"); ax.spines[:].set_visible(False); ax.tick_params(length=0)
    ax.set_title("SHAP importance rank (1 = most important), 7-region test set", loc="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig2_shap_rank_by_seed.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_demo_vs_full():
    full = pd.read_csv(RES / "shap_rank_test_full.csv", index_col=0)
    demo = pd.read_csv(RES / "shap_rank_test_demo.csv", index_col=0)
    order = full.median(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(order))
    for k, (tab, col, lab, off) in enumerate([(full, BLUE, "7-region test set", -0.13),
                                              (demo, ORANGE, "Paper's 2-region test set", 0.13)]):
        med = tab.loc[order].median(axis=1); lo = tab.loc[order].min(axis=1); hi = tab.loc[order].max(axis=1)
        ax.hlines(y + off, lo, hi, color=col, linewidth=2, alpha=0.45)
        ax.scatter(med, y + off, color=col, s=46, edgecolor=SURF, linewidth=1.5, zorder=3, label=lab)
    ax.set_yticks(y, order); ax.invert_yaxis(); ax.set_xlim(0.5, 12.5); ax.set_xticks(range(1, 13))
    ax.set_xlabel("SHAP rank across 10 seeds (dot = median, line = min to max)")
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig3_rank_depends_on_test_set.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_column_order():
    pairs = [("USFLUX", "AREA_ACR", 0.92, -0.11), ("TOTUSJH", "TOTUSJZ", 0.98, -0.12),
             ("MEANSHR", "SHRGT45", 0.99, -0.22), ("MEANJZH", "MEANALP", 0.98, 0.43)]
    txt = (RES / "column_order_check.txt").read_text()
    vals = []
    for a, b, _, _ in pairs:
        line = [ln for ln in txt.splitlines() if ln.strip().startswith(a) and f"vs {b}" in ln][0]
        nums = [float(x.split("=")[1].split()[0]) for x in line.split("|")]
        vals.append(nums)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(pairs)); w = 0.36
    hv = [v[0] for v in vals]; pv = [v[1] for v in vals]
    ax.bar(x - w / 2 - 0.01, hv, w, color=BLUE, label="Headered files (CME_data_samples/)")
    ax.bar(x + w / 2 + 0.01, pv, w, color=ORANGE, label="Code-folder files read in ReadMe order")
    for xi, v in zip(x - w / 2, hv): ax.text(xi, v + 0.04, f"{v:+.2f}", ha="center", fontsize=8.5, color=INK)
    for xi, v in zip(x + w / 2, pv):
        ax.text(xi, v + (0.04 if v >= 0 else -0.12), f"{v:+.2f}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x, [f"{a}\nvs {b}" for a, b, _, _ in pairs]); ax.set_ylim(-0.45, 1.2)
    ax.set_ylabel("Pearson r (training rows)")
    ax.set_title("Pairs of SHARP parameters that physically move together", loc="left", color=INK, fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "fig4_column_order_check.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_tss(); fig_rank_heatmap(); fig_demo_vs_full(); fig_column_order()
    print("figures written to", FIG)
