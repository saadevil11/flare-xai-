"""SHAP stability analysis.

1. GradientExplainer SHAP (as in Rawashdeh, Wang & Herbert, FLAIRS 2025) for each seed.
2. How stable is the global feature ranking across seeds and across test sets?
3. Correlation clusters among the 12 SHARP features, and grouped attributions.
4. Permutation importance per feature vs. per correlated group.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from data import FEATURES, SERIES_LEN, load_all, read  # noqa: E402
from model import LSTMAttention  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CORR_THRESHOLD = 0.9


def load_model(seed, features=FEATURES, tag="lstm"):
    m = LSTMAttention(len(features), SERIES_LEN)
    m.load_state_dict(torch.load(RES / "models" / f"{tag}_seed{seed}.pt"))
    return m.eval()


def shap_values(model, background, X, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    ex = shap.GradientExplainer(model, torch.tensor(background), batch_size=500)
    sv = ex.shap_values(torch.tensor(X), nsamples=200, rseed=seed)
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
    return sv.reshape(X.shape)                      # (n, T, F)


def correlation_groups(features=FEATURES):
    df = read("normalized_training_24.csv")
    df = df[(df[features] != 0).all(axis=1)]
    corr = df[features].corr().abs()
    dist = squareform(1 - corr.to_numpy(), checks=False)
    labels = fcluster(linkage(dist, "average"), t=1 - CORR_THRESHOLD, criterion="distance")
    groups = {}
    for f, g in zip(features, labels):
        groups.setdefault(int(g), []).append(f)
    named = {"+".join(v): v for v in groups.values()}
    return corr, named


def ranking(values: pd.Series):
    return values.rank(ascending=False, method="min").astype(int)


def pairwise_tau(rank_table: pd.DataFrame):
    taus = [kendalltau(rank_table[a], rank_table[b])[0] for a, b in combinations(rank_table.columns, 2)]
    return float(np.mean(taus)), float(np.min(taus)), float(np.max(taus))


def permutation_auc_drop(model, X, y, cols_idx, rng, repeats=5):
    with torch.no_grad():
        base = roc_auc_score(y, model(torch.tensor(X)).squeeze(1).numpy())
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, :, cols_idx] = X[perm][:, :, cols_idx]   # permute whole sequences jointly
            drops.append(base - roc_auc_score(y, model(torch.tensor(Xp)).squeeze(1).numpy()))
    return float(np.mean(drops))


def main(seeds=tuple(range(10))):
    d = load_all()
    Xtr, ytr, _ = d["train"]
    rng = np.random.default_rng(123)
    background = Xtr[rng.choice(len(Xtr), 200, replace=False)]
    corr, groups = correlation_groups()
    corr.to_csv(RES / "feature_correlation_train.csv")

    feat_imp, group_imp = {}, {}
    perm_feat, perm_group = {}, {}
    for s in seeds:
        m = load_model(s)
        for key in ("test_full", "test_demo"):
            X, y, _ = d[key]
            sv = shap_values(m, background, X, seed=s)
            per_feat_signed = sv.sum(axis=1)                  # (n, F): total contribution of each feature
            feat_imp[(key, s)] = pd.Series(np.abs(per_feat_signed).mean(0), index=FEATURES)
            group_imp[(key, s)] = pd.Series(
                {g: np.abs(per_feat_signed[:, [FEATURES.index(f) for f in fs]].sum(1)).mean()
                 for g, fs in groups.items()})
        X, y, _ = d["test_full"]
        prng = np.random.default_rng(s)
        perm_feat[s] = pd.Series({f: permutation_auc_drop(m, X, y, [i], prng) for i, f in enumerate(FEATURES)})
        perm_group[s] = pd.Series({g: permutation_auc_drop(m, X, y, [FEATURES.index(f) for f in fs], prng)
                                   for g, fs in groups.items()})
        print(f"seed {s} done; top SHAP feature on full test: {feat_imp[('test_full', s)].idxmax()}")

    summary = {"groups": groups, "corr_threshold": CORR_THRESHOLD}
    tables = {}
    for key in ("test_full", "test_demo"):
        fi = pd.DataFrame({f"seed{s}": feat_imp[(key, s)] for s in seeds})
        gi = pd.DataFrame({f"seed{s}": group_imp[(key, s)] for s in seeds})
        fr, gr = fi.apply(ranking), gi.apply(ranking)
        fi.to_csv(RES / f"shap_importance_{key}.csv"); fr.to_csv(RES / f"shap_rank_{key}.csv")
        gi.to_csv(RES / f"shap_group_importance_{key}.csv"); gr.to_csv(RES / f"shap_group_rank_{key}.csv")
        tables[key] = (fi, fr, gi, gr)
        summary[key] = {
            "feature_rank_tau_mean_min_max": pairwise_tau(fr),
            "group_rank_tau_mean_min_max": pairwise_tau(gr),
            "top1_feature_per_seed": fi.idxmax().to_dict(),
            "top3_features_per_seed": {c: list(fi[c].sort_values(ascending=False).index[:3]) for c in fi},
            "top1_group_per_seed": gi.idxmax().to_dict(),
            "feature_rank_range": {f: [int(fr.loc[f].min()), int(fr.loc[f].max())] for f in FEATURES},
        }
    # same model, different test set: does the explanation depend on which regions are explained?
    full_r, demo_r = tables["test_full"][1], tables["test_demo"][1]
    summary["demo_vs_full_tau_per_seed"] = {c: float(kendalltau(full_r[c], demo_r[c])[0]) for c in full_r}

    pf = pd.DataFrame({f"seed{s}": perm_feat[s] for s in seeds})
    pg = pd.DataFrame({f"seed{s}": perm_group[s] for s in seeds})
    pf.to_csv(RES / "permutation_auc_drop_features.csv"); pg.to_csv(RES / "permutation_auc_drop_groups.csv")
    summary["permutation_mean_auc_drop_features"] = pf.mean(1).sort_values(ascending=False).round(4).to_dict()
    summary["permutation_mean_auc_drop_groups"] = pg.mean(1).sort_values(ascending=False).round(4).to_dict()
    (RES / "shap_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
