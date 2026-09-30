"""
Pre-Emblem variable inspection and banding.

Workflow:
    1. profile_numeric / one_way      -> inspect distribution and exposure-weighted response
    2. quantile_edges / tree_edges    -> propose candidate cut points
    3. merge_to_credibility           -> merge thin bands to a minimum claim count
    4. round edges by hand            -> business-sensible cut points (21, 25, 30...)
    5. record in BANDS spec           -> single source of truth, saved to JSON
    6. apply_spec                     -> banded integer codes for Emblem + label lookups
    7. validate                       -> no unassigned rows, exposure reconciles

Conventions:
    - Numeric bands are lower-inclusive [a, b) by default, so edges [21, 25] give
      "<21", "21-24", "25+" for integer variables.
    - Code 0 is always the Missing band; codes 1..k follow natural order.
    - Categorical levels below an exposure threshold go to an "Other" level.
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeRegressor

MISSING_CODE = 0


# ---------------------------------------------------------------- inspection

def profile_numeric(df, col, expo="exposure"):
    """Missing rate, cardinality and exposure-weighted percentiles."""
    s, w = df[col], df[expo]
    m = s.notna()
    qs = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    wq = _weighted_quantiles(s[m].to_numpy(), w[m].to_numpy(), qs)
    return pd.Series(
        {
            "rows": len(s),
            "missing_rows": int((~m).sum()),
            "missing_expo_share": w[~m].sum() / w.sum(),
            "distinct": s.nunique(),
            **{f"wq{int(q * 100):02d}": v for q, v in zip(qs, wq)},
        },
        name=col,
    )


def one_way(df, col, expo="exposure", claims="claim_count", cost=None):
    """Exposure-weighted one-way table. Missing kept as its own row."""
    agg = {"exposure": (expo, "sum"), "claims": (claims, "sum"), "rows": (expo, "size")}
    if cost:
        agg["cost"] = (cost, "sum")
    g = df.groupby(col, dropna=False, observed=False).agg(**agg)
    g["expo_share"] = g.exposure / g.exposure.sum()
    g["freq"] = g.claims / g.exposure
    g["freq_rel"] = g.freq / (g.claims.sum() / g.exposure.sum())
    if cost:
        g["sev"] = g.cost / g.claims.replace(0, np.nan)
        g["bc"] = g.cost / g.exposure
    return g


def plot_one_way(tbl, title="", response="freq"):
    """Exposure bars + response line. Uses make_subplots secondary_y (not yaxis='y2')."""
    x = [str(i) for i in tbl.index]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x, y=tbl.exposure, name="Exposure", opacity=0.4), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=tbl[response], name=response, mode="lines+markers"), secondary_y=True)
    fig.update_layout(title=title, xaxis_type="category")
    fig.update_yaxes(title_text="Exposure", secondary_y=False)
    fig.update_yaxes(title_text=response, secondary_y=True)
    return fig


def stability(df, col, period, expo="exposure", claims="claim_count"):
    """Frequency relativity by level x period — checks a pattern holds across years."""
    g = df.groupby([period, col], dropna=False, observed=False)[[expo, claims]].sum()
    g["freq"] = g[claims] / g[expo]
    overall = g.groupby(level=0)[claims].sum() / g.groupby(level=0)[expo].sum()
    g["freq_rel"] = g["freq"] / overall.reindex(g.index.get_level_values(0)).to_numpy()
    return g["freq_rel"].unstack(0)


def cramers_v(df, a, b):
    """Association between two (banded/categorical) factors — flags confounding."""
    ct = pd.crosstab(df[a].fillna("NA"), df[b].fillna("NA"))
    n = ct.to_numpy().sum()
    exp = np.outer(ct.sum(1), ct.sum(0)) / n
    chi2 = ((ct.to_numpy() - exp) ** 2 / exp).sum()
    r, k = ct.shape
    return np.sqrt(chi2 / n / max(1, min(r - 1, k - 1)))


# ------------------------------------------------------- candidate cut points

def _weighted_quantiles(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cw, x)


def quantile_edges(df, col, n_bands, expo="exposure"):
    """Equal-EXPOSURE bands (not equal row count)."""
    m = df[col].notna()
    qs = np.linspace(0, 1, n_bands + 1)[1:-1]
    return np.unique(_weighted_quantiles(df.loc[m, col].to_numpy(), df.loc[m, expo].to_numpy(), qs))


def tree_edges(df, col, expo="exposure", claims="claim_count", max_bands=12, min_expo_frac=0.02):
    """
    Response-driven split suggestions: Poisson tree on claim rate, weighted by exposure.
    One-way only -> treat as suggestions, not final bands (see note on confounding).
    """
    m = df[col].notna() & (df[expo] > 0)
    X = df.loc[m, [col]].to_numpy()
    y = (df.loc[m, claims] / df.loc[m, expo]).to_numpy()
    w = df.loc[m, expo].to_numpy()
    t = DecisionTreeRegressor(
        criterion="poisson", max_leaf_nodes=max_bands, min_weight_fraction_leaf=min_expo_frac
    ).fit(X, y, sample_weight=w)
    return np.sort(t.tree_.threshold[t.tree_.feature >= 0])


def merge_to_credibility(df, col, edges, min_claims, claims="claim_count", right=False):
    """Repeatedly merge the thinnest band into its thinner neighbour until all bands >= min_claims."""
    edges = sorted(edges)
    while edges:
        b = pd.cut(df[col], [-np.inf, *edges, np.inf], right=right)
        c = df.groupby(b, observed=False)[claims].sum().to_numpy()
        i = int(c.argmin())
        if c[i] >= min_claims:
            break
        if i == 0:
            j = 0                                   # merge with band 1
        elif i == len(c) - 1:
            j = i - 1                               # merge with band i-1
        else:
            j = i - 1 if c[i - 1] <= c[i + 1] else i
        edges.pop(j)                                # edge j separates bands j and j+1
    return edges


# ------------------------------------------------------------- band spec

def numeric_labels(edges, right=False, integer=True, fmt="{:g}"):
    """Human-readable labels matching pd.cut intervals."""
    e = list(edges)
    if right:   # (a, b]
        lab = [f"<={fmt.format(e[0])}"] + [f"{fmt.format(a)}<x<={fmt.format(b)}" for a, b in zip(e, e[1:])] + [f">{fmt.format(e[-1])}"]
    elif integer:  # [a, b) on integers -> a to b-1
        lab = [f"<{fmt.format(e[0])}"] + [f"{fmt.format(a)}-{fmt.format(b - 1)}" for a, b in zip(e, e[1:])] + [f"{fmt.format(e[-1])}+"]
    else:
        lab = [f"<{fmt.format(e[0])}"] + [f"{fmt.format(a)}<=x<{fmt.format(b)}" for a, b in zip(e, e[1:])] + [f">={fmt.format(e[-1])}"]
    return lab


def categorical_map(df, col, min_expo_share=0.005, expo="exposure", other="OTHER"):
    """Levels under the exposure threshold -> OTHER. Returns {raw_level: grouped_level}."""
    share = df.groupby(col, observed=True)[expo].sum() / df[expo].sum()
    return {str(k): (str(k) if v >= min_expo_share else other) for k, v in share.items()}


def apply_band(s, spec):
    """Return integer codes: 0 = Missing, 1..k in natural order."""
    if spec["type"] == "numeric":
        codes = pd.cut(s, [-np.inf, *spec["edges"], np.inf], right=spec.get("right", False), labels=False)
        return (codes + 1).fillna(MISSING_CODE).astype(int)
    if spec["type"] == "categorical":
        levels = spec["levels"]                     # ordered list of grouped levels
        other = spec.get("other", "OTHER")
        grouped = s.astype("string").map(spec["map"]).fillna(other)   # unseen levels -> OTHER
        grouped = grouped.where(s.notna(), None)
        lookup = {lvl: i + 1 for i, lvl in enumerate(levels)}
        return grouped.map(lookup).fillna(MISSING_CODE).astype(int)
    raise ValueError(spec["type"])


def label_table(name, spec):
    """Code -> label lookup, for Emblem level labels and Radar rating tables."""
    if spec["type"] == "numeric":
        labs = numeric_labels(spec["edges"], spec.get("right", False), spec.get("integer", True))
        rows = [(MISSING_CODE, "Missing", None, None)]
        lows = [-np.inf, *spec["edges"]]
        highs = [*spec["edges"], np.inf]
        rows += [(i + 1, l, lo, hi) for i, (l, lo, hi) in enumerate(zip(labs, lows, highs))]
        return pd.DataFrame(rows, columns=["code", "label", "lower", "upper"]).assign(factor=name)
    rows = [(MISSING_CODE, "Missing")] + [(i + 1, l) for i, l in enumerate(spec["levels"])]
    return pd.DataFrame(rows, columns=["code", "label"]).assign(factor=name)


def apply_spec(df, bands, suffix="_band", keep_raw=True):
    out = df.copy()
    for name, spec in bands.items():
        out[name + suffix] = apply_band(df[spec.get("source", name)], spec)
    if not keep_raw:
        out = out.drop(columns=[s.get("source", n) for n, s in bands.items()])
    return out


def save_spec(bands, path, version):
    with open(path, "w") as f:
        json.dump({"version": version, "bands": bands}, f, indent=2, default=float)


def load_spec(path):
    with open(path) as f:
        return json.load(f)["bands"]


# ---------------------------------------------------------------- checks

def validate(raw, banded, bands, expo="exposure", suffix="_band", min_claims=None, claims="claim_count"):
    """Row count, exposure reconciliation, no NaN codes, thin-band warnings."""
    assert len(raw) == len(banded), "row count changed"
    assert np.isclose(raw[expo].sum(), banded[expo].sum()), "exposure does not reconcile"
    report = {}
    for name in bands:
        c = banded[name + suffix]
        assert c.notna().all(), f"{name}: unassigned rows"
        t = banded.groupby(c)[[expo, claims]].sum()
        report[name] = t
        if min_claims:
            thin = t[t[claims] < min_claims]
            if len(thin):
                print(f"WARNING {name}: codes {list(thin.index)} below {min_claims} claims")
    return report
