import numpy as np


def confusion(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fn, fp, tn


def tss(y, pred):
    """True Skill Statistic = recall(P) - false-positive rate. Range [-1, 1]."""
    tp, fn, fp, tn = confusion(y, pred)
    if tp + fn == 0 or fp + tn == 0:
        return np.nan
    return tp / (tp + fn) - fp / (fp + tn)


def hss(y, pred):
    """Heidke Skill Score (HSS2), as used in flare-forecasting papers."""
    tp, fn, fp, tn = confusion(y, pred)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return np.nan if den == 0 else 2 * (tp * tn - fn * fp) / den


def region_bootstrap_tss(y, pred, regions, n_boot=2000, seed=0):
    """95% interval for TSS when whole active regions (not samples) are resampled.

    Samples within one region are 12 minutes apart and highly correlated, so the
    region is the realistic unit of independence.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pred = np.asarray(pred); regions = np.asarray(regions)
    uniq = np.unique(regions)
    idx = {r: np.where(regions == r)[0] for r in uniq}
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[r] for r in rng.choice(uniq, size=len(uniq), replace=True)])
        v = tss(y[pick], pred[pick])
        if not np.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)
