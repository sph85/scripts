"""
Decline driver analysis - Commercial Property Owners
====================================================

Finds which rating/risk features most strongly drive a quote being declined,
using permutation importance on a held-out set.

Pipeline
--------
1. Load quote-level data (one row per quote, binary `declined` target).
2. Fit a gradient-boosted classifier (sklearn HistGradientBoosting - native
   categoricals and missing values, no one-hot needed).
3. Permutation importance on the TEST set, scored on PR-AUC (decline is usually
   rare, so ROC-AUC flatters) and log loss.
4. Grouped permutation importance: correlated features (e.g. sum insured /
   floor area / rebuild cost) are permuted together, because single-feature
   permutation understates features that have a close substitute.
5. Shallow surrogate decision tree - readable rules you can hold up against
   the actual underwriting referral/decline guide.
6. Decline rate by band/level for the top drivers as a sanity check.

Usage
-----
    python decline_drivers.py --demo                    # synthetic data, runs end to end
    python decline_drivers.py --data quotes.parquet     # your data

On Databricks, replace `load_data()` with something like:
    spark.table("pricing.cpo_quotes").filter(...).toPandas()

Requires: scikit-learn >= 1.4, pandas, numpy, scipy, matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, log_loss, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

# --------------------------------------------------------------------------
# CONFIG - edit for your data
# --------------------------------------------------------------------------
TARGET = "declined"            # 1 = declined, 0 = quoted (exclude NTUs/referred-then-quoted as you see fit)
DATE_COL = "quote_date"        # used for an out-of-time split; set None for a random stratified split
ID_COLS = ["quote_id", "policy_id"]

# Anything populated AT or AFTER the decision leaks the answer. Drop it.
LEAKAGE_COLS = [
    "decline_reason", "referral_flag", "referral_reason", "underwriter_id",
    "quoted_premium", "technical_premium", "commission", "quote_status",
]

# Optional: hand-specified feature groups to permute together. Anything not
# listed here gets grouped automatically by correlation clustering.
MANUAL_GROUPS: dict[str, list[str]] = {
    # "size": ["sum_insured", "floor_area", "rebuild_cost"],
    # "location_peril": ["flood_zone", "subsidence_score", "postcode_area"],
}

CORR_CLUSTER_THRESHOLD = 0.7   # |Spearman| above ~this => same auto group
MAX_CAT_LEVELS = 250           # HGB limit is 255 incl. missing; rarer levels -> "OTHER"
MIN_CAT_COUNT = 30             # levels with fewer rows than this -> "OTHER"
N_REPEATS = 10                 # permutation repeats (more = tighter intervals)
TEST_SIZE = 0.25
RANDOM_STATE = 42
OUT_DIR = Path("decline_driver_outputs")


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)


def make_demo_data(n: int = 25_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Synthetic CPO quotes with known, rule-like decline logic."""
    rng = np.random.default_rng(seed)
    floor_area = rng.lognormal(6.5, 0.8, n)
    sum_insured = floor_area * rng.normal(1800, 250, n)
    df = pd.DataFrame({
        "quote_id": np.arange(n),
        "quote_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 900, n), "D"),
        "floor_area": floor_area.round(),
        "sum_insured": sum_insured.round(-3),
        "rebuild_cost": (sum_insured * rng.normal(0.95, 0.05, n)).round(-3),
        "building_age": rng.gamma(2.5, 25, n).round(),
        "construction": rng.choice(["standard", "non_standard", "timber_frame", "listed"], n, p=[.75, .12, .08, .05]),
        "occupancy": rng.choice(["office", "retail", "industrial", "mixed_residential", "unoccupied"], n, p=[.3, .3, .2, .15, .05]),
        "tenant_trade": rng.choice(["office", "shop", "restaurant", "takeaway", "workshop", "nightclub"], n, p=[.3, .3, .15, .1, .12, .03]),
        "flood_zone": rng.choice(["1", "2", "3"], n, p=[.8, .13, .07]),
        "claims_5yr": rng.poisson(0.4, n),
        "sprinklered": rng.choice([0, 1], n, p=[.85, .15]),
        "roof_flat_pct": rng.choice([0, 10, 25, 50, 100], n, p=[.5, .2, .15, .1, .05]),
        "years_trading": rng.gamma(2, 6, n).round(),
        "postcode_area": rng.choice([f"PC{i:02d}" for i in range(60)], n),
        "quoted_premium": rng.lognormal(7, 0.5, n),     # leakage col - should be dropped
    })
    df.loc[rng.random(n) < 0.05, "building_age"] = np.nan

    logit = (
        -4.2
        + 3.5 * (df.occupancy == "unoccupied")
        + 2.0 * ((df.flood_zone == "3") & (df.claims_5yr >= 1))
        + 0.8 * (df.flood_zone == "3")
        + 1.8 * ((df.construction == "non_standard") & (df.sum_insured > 2.5e6))
        + 1.5 * (df.tenant_trade.isin(["nightclub", "takeaway"]) & (df.sprinklered == 0))
        + 0.6 * df.claims_5yr
        + 1.2 * (df.roof_flat_pct >= 50)
        + 0.01 * df.building_age.fillna(60).clip(upper=200)
    )
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    drop = [c for c in ID_COLS + LEAKAGE_COLS if c in df.columns]
    if drop:
        print(f"Dropping ID/leakage columns: {drop}")
    dates = df[DATE_COL] if DATE_COL and DATE_COL in df.columns else None
    y = df[TARGET].astype(int)
    X = df.drop(columns=drop + [TARGET] + ([DATE_COL] if dates is not None else []))

    for c in X.columns:
        # Anything non-numeric (object, pandas>=3 'str', category, bool) -> category
        if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            s = X[c].astype("string")
            counts = s.value_counts()
            keep = counts[counts >= MIN_CAT_COUNT].index[: MAX_CAT_LEVELS - 1]
            s = s.where(s.isin(keep) | s.isna(), "OTHER")
            X[c] = s.astype("category")
    return X, y, dates


def split(X, y, dates):
    if dates is not None:
        cutoff = dates.quantile(1 - TEST_SIZE)
        tr, te = dates < cutoff, dates >= cutoff
        print(f"Out-of-time split at {pd.Timestamp(cutoff).date()}: train {tr.sum():,} / test {te.sum():,}")
        return X[tr], X[te], y[tr], y[te]
    print("Random stratified split")
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------
def fit_model(X_tr, y_tr) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        learning_rate=0.05,
        max_iter=1000,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        scoring="loss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr, y_tr)
    print(f"Boosting iterations used: {model.n_iter_}")
    return model


def evaluate(model, X, y, label: str) -> None:
    p = model.predict_proba(X)[:, 1]
    print(f"[{label}] decline rate {y.mean():.2%} | ROC-AUC {roc_auc_score(y, p):.3f} | "
          f"PR-AUC {average_precision_score(y, p):.3f} (baseline {y.mean():.3f}) | "
          f"log loss {log_loss(y, p):.4f}")


# --------------------------------------------------------------------------
# IMPORTANCE
# --------------------------------------------------------------------------
def single_feature_importance(model, X_te, y_te) -> pd.DataFrame:
    res = permutation_importance(
        model, X_te, y_te,
        scoring=["average_precision", "neg_log_loss"],
        n_repeats=N_REPEATS, random_state=RANDOM_STATE, n_jobs=-1,
    )
    out = pd.DataFrame({
        "feature": X_te.columns,
        "pr_auc_drop": res["average_precision"].importances_mean,
        "pr_auc_drop_sd": res["average_precision"].importances_std,
        "logloss_increase": res["neg_log_loss"].importances_mean,
        "logloss_increase_sd": res["neg_log_loss"].importances_std,
    }).sort_values("pr_auc_drop", ascending=False, ignore_index=True)
    # A feature is only meaningfully important if its drop clears ~2 SDs of noise
    out["significant"] = out["pr_auc_drop"] > 2 * out["pr_auc_drop_sd"]
    return out


def correlation_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    """Cluster features by |Spearman| (categoricals via their codes - a rough
    association measure, fine for spotting obvious duplicates)."""
    manual = {f for g in MANUAL_GROUPS.values() for f in g}
    cols = [c for c in X.columns if c not in manual]
    groups = dict(MANUAL_GROUPS)
    if len(cols) < 2:
        return groups | {c: [c] for c in cols}

    num = pd.DataFrame({
        c: (X[c].cat.codes.replace(-1, np.nan) if isinstance(X[c].dtype, pd.CategoricalDtype) else X[c])
        for c in cols
    }).astype(float)
    num = num.fillna(num.median())
    corr = spearmanr(num).correlation
    corr = np.nan_to_num((corr + corr.T) / 2)
    np.fill_diagonal(corr, 1.0)
    dist = squareform(1 - np.abs(corr), checks=False)
    labels = hierarchy.fcluster(hierarchy.linkage(dist, "average"), 1 - CORR_CLUSTER_THRESHOLD, "distance")

    for lab in np.unique(labels):
        members = [c for c, l in zip(cols, labels) if l == lab]
        groups[" + ".join(members) if len(members) > 1 else members[0]] = members
    return groups


def grouped_importance(model, X_te, y_te, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Permute every column in a group with the SAME row shuffle, so the group's
    joint distribution is kept but its link to the target is broken."""
    rng = np.random.default_rng(RANDOM_STATE)
    base = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(N_REPEATS):
            idx = rng.permutation(len(X_te))
            Xp = X_te.copy()
            for c in cols:
                Xp[c] = X_te[c].iloc[idx].to_numpy()
                if isinstance(X_te[c].dtype, pd.CategoricalDtype):
                    Xp[c] = pd.Categorical(Xp[c], categories=X_te[c].cat.categories)
            drops.append(base - average_precision_score(y_te, model.predict_proba(Xp)[:, 1]))
        rows.append({"group": name, "n_features": len(cols),
                     "pr_auc_drop": np.mean(drops), "pr_auc_drop_sd": np.std(drops)})
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------
# INTERPRETATION AIDS
# --------------------------------------------------------------------------
def surrogate_rules(model, X_tr, max_depth: int = 4) -> str:
    """Shallow tree fitted to the GBM's predictions - readable decline 'rules'."""
    Xd = X_tr.copy()
    for c in Xd.columns:
        if isinstance(Xd[c].dtype, pd.CategoricalDtype):
            Xd[c] = Xd[c].cat.add_categories("MISSING").fillna("MISSING") if "MISSING" not in Xd[c].cat.categories else Xd[c].fillna("MISSING")
        else:
            Xd[c] = Xd[c].fillna(-999)
    Xd = pd.get_dummies(Xd, prefix_sep="=", dtype=float)
    # Regress on the GBM's predicted decline probability, so each leaf shows a
    # decline RATE rather than a hard 0/1 class (more informative when declines are rare)
    p = model.predict_proba(X_tr)[:, 1]
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=200, random_state=RANDOM_STATE)
    tree.fit(Xd, p)
    fidelity = r2_score(p, tree.predict(Xd))
    header = (f"Surrogate tree R^2 vs GBM predicted decline probability: {fidelity:.1%}\n"
              f"(leaf 'value' = mean predicted decline probability; one-hot splits read '<= 0' = NOT that level)\n")
    return header + export_text(tree, feature_names=list(Xd.columns), decimals=3)


def decline_rate_tables(X, y, features: list[str], bins: int = 5) -> pd.DataFrame:
    out = []
    for f in features:
        s = X[f]
        if not isinstance(s.dtype, pd.CategoricalDtype) and s.nunique() > 10:
            s = pd.qcut(s, bins, duplicates="drop")
        t = (pd.DataFrame({"level": s.astype(str), "y": y.values})
             .groupby("level", observed=True)["y"].agg(quotes="size", decline_rate="mean")
             .reset_index())
        t.insert(0, "feature", f)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def plot_importance(imp: pd.DataFrame, label_col: str, path: Path, top: int = 20) -> None:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(d) + 1.5))
    ax.barh(d[label_col], d["pr_auc_drop"], xerr=d["pr_auc_drop_sd"], color="#3b6ea5", ecolor="#555")
    ax.set_xlabel("Drop in PR-AUC when permuted (test set)")
    ax.set_title("Decline drivers - permutation importance")
    ax.axvline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to quote-level CSV or parquet")
    ap.add_argument("--demo", action="store_true", help="Run on synthetic data")
    args = ap.parse_args()
    if not (args.data or args.demo):
        ap.error("pass --data PATH or --demo")

    OUT_DIR.mkdir(exist_ok=True)
    df = make_demo_data() if args.demo else load_data(args.data)
    X, y, dates = prepare(df)
    X_tr, X_te, y_tr, y_te = split(X, y, dates)

    model = fit_model(X_tr, y_tr)
    evaluate(model, X_tr, y_tr, "train")
    evaluate(model, X_te, y_te, "test ")

    print("\nSingle-feature permutation importance (test set)")
    imp = single_feature_importance(model, X_te, y_te)
    print(imp.round(4).to_string(index=False))
    imp.to_csv(OUT_DIR / "importance_single.csv", index=False)
    plot_importance(imp, "feature", OUT_DIR / "importance_single.png")

    groups = correlation_groups(X_tr)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        print("\nCorrelated groups permuted together:")
        for k, v in multi.items():
            print(f"  {v}")
    gimp = grouped_importance(model, X_te, y_te, groups)
    print("\nGrouped permutation importance (test set)")
    print(gimp.round(4).to_string(index=False))
    gimp.to_csv(OUT_DIR / "importance_grouped.csv", index=False)
    plot_importance(gimp, "group", OUT_DIR / "importance_grouped.png")

    rules = surrogate_rules(model, X_tr)
    (OUT_DIR / "surrogate_rules.txt").write_text(rules)
    print("\n" + rules)

    top = imp.loc[imp["significant"], "feature"].head(8).tolist()
    rates = decline_rate_tables(X, y, top)
    rates.to_csv(OUT_DIR / "decline_rates_top_features.csv", index=False)
    print(f"Outputs written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
