"""Train the LSTM-attention model with several random seeds and evaluate it on
(a) the lab's 540-sample demo test set (2 active regions) and
(b) the full 2016-2017 test set (7 active regions).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent))
from data import FEATURES, SERIES_LEN, load_all, read  # noqa: E402
from metrics import hss, region_bootstrap_tss, tss  # noqa: E402
from model import LSTMAttention  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; MODELS = RES / "models"
THRESH = 0.45           # lab's threshold for lstm / 24 h
EPOCHS, BATCH = 20, 256  # lab's settings
torch.set_num_threads(max(1, torch.get_num_threads()))


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def train_one(X, y, seed, features=FEATURES):
    set_seed(seed)
    model = LSTMAttention(len(features), SERIES_LEN)
    w = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=y)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9, eps=1e-7)  # Keras defaults
    Xt, yt = torch.tensor(X), torch.tensor(y)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for b in range(0, len(Xt), BATCH):
            i = perm[b:b + BATCH]
            p = model(Xt[i]).squeeze(1).clamp(1e-7, 1 - 1e-7)
            weight = torch.where(yt[i] == 1, w[1], w[0]).float()
            loss = torch.nn.functional.binary_cross_entropy(p, yt[i], weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, X):
    return model(torch.tensor(X)).squeeze(1).numpy()


def evaluate(prob, y, meta, name):
    pred = (prob >= THRESH).astype(int)
    lo, hi, nb = region_bootstrap_tss(y, pred, meta["NOAA_AR_NUM"].to_numpy())
    per_region = (pd.DataFrame({"region": meta["NOAA_AR_NUM"], "y": y, "pred": pred})
                  .groupby("region").agg(n=("y", "size"), frac_true_P=("y", "mean"), frac_pred_P=("pred", "mean")))
    return {"set": name, "n": int(len(y)), "regions": int(meta["NOAA_AR_NUM"].nunique()),
            "TSS": tss(y, pred), "HSS": hss(y, pred), "TSS_region_boot_lo": lo, "TSS_region_boot_hi": hi,
            "valid_boot": nb}, per_region


def main(seeds=(0, 1, 2, 3, 4), features=FEATURES, tag="lstm"):
    RES.mkdir(exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    d = load_all(features)
    Xtr, ytr, _ = d["train"]
    print("train", Xtr.shape, "pos rate", ytr.mean().round(3))
    rows, regions = [], []
    for s in seeds:
        m = train_one(Xtr, ytr, s, features)
        torch.save(m.state_dict(), MODELS / f"{tag}_seed{s}.pt")
        for key in ("test_demo", "test_full"):
            X, y, meta = d[key]
            r, pr = evaluate(predict(m, X), y, meta, key)
            r.update(seed=s, model=tag); rows.append(r)
            pr = pr.reset_index(); pr["seed"] = s; pr["set"] = key; pr["model"] = tag; regions.append(pr)
            print(f"seed {s} {key}: TSS={r['TSS']:.3f} HSS={r['HSS']:.3f} "
                  f"region-bootstrap 95% [{r['TSS_region_boot_lo']:.2f}, {r['TSS_region_boot_hi']:.2f}]")
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"metrics_{tag}.csv", index=False)
    pd.concat(regions).to_csv(RES / f"per_region_{tag}.csv", index=False)
    return out


def baselines():
    """Logistic regression on the last time step, plus the lab's own saved predictions."""
    d = load_all()
    Xtr, ytr, _ = d["train"]
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr[:, -1, :], ytr)
    rows = []
    for key in ("test_demo", "test_full"):
        X, y, meta = d[key]
        r, _ = evaluate(lr.predict_proba(X[:, -1, :])[:, 1], y, meta, key)
        r.update(model="logreg_last_step", seed=-1); rows.append(r)
    lab = pd.read_csv(ROOT / "data" / "lab_lstm-24-output.csv")
    yl = (lab["Label"] == "P").astype(int).to_numpy(); pl = (lab["Predicted Label"] == "P").astype(int).to_numpy()
    lo, hi, nb = region_bootstrap_tss(yl, pl, lab["NOAA AR NUM"].to_numpy())
    rows.append({"set": "test_demo", "n": len(yl), "regions": int(lab["NOAA AR NUM"].nunique()),
                 "TSS": tss(yl, pl), "HSS": hss(yl, pl), "TSS_region_boot_lo": lo, "TSS_region_boot_hi": hi,
                 "valid_boot": nb, "model": "lab_saved_lstm24", "seed": -1})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "metrics_baselines.csv", index=False)
    print(out[["model", "set", "TSS", "HSS"]])
    return out


def reevaluate(seeds=tuple(range(10)), features=FEATURES, tag="lstm"):
    """Score already-trained models again (no retraining)."""
    d = load_all(features)
    rows, regions = [], []
    for s in seeds:
        m = LSTMAttention(len(features), SERIES_LEN)
        m.load_state_dict(torch.load(MODELS / f"{tag}_seed{s}.pt")); m.eval()
        for key in ("test_demo", "test_full"):
            X, y, meta = d[key]
            r, pr = evaluate(predict(m, X), y, meta, key)
            r.update(seed=s, model=tag); rows.append(r)
            pr = pr.reset_index(); pr["seed"] = s; pr["set"] = key; pr["model"] = tag; regions.append(pr)
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"metrics_{tag}.csv", index=False)
    pd.concat(regions).to_csv(RES / f"per_region_{tag}.csv", index=False)
    return out


if __name__ == "__main__":
    baselines()
    main(seeds=tuple(range(10)))
