"""
Pre-Emblem variable review and banding.

Main entry points
-----------------
load_data(path, columns)          read .sas7bdat / .xpt / .csv / .parquet (column names case-insensitive)
prepare(df, ...)                  blank claims -> 0, drop zero exposure, floor negative cost
raw_view_table / plot_raw         every value (or ~100 unrounded bins) with a smoothed trend
peril_overview(...)               per factor: burning cost relativity by peril + peril mix
review_all(df, factors, ...)      loop over every factor: profile, charts, suggested bands
                                  (show=True: inline in Jupyter; report=HtmlReport(): HTML file)
review_variable(df, col, ...)     the same for one factor (use with spec=... to check an override)
apply_spec / validate / save_spec / label_table
                                  build the Emblem file, check it, and export the band definitions

Usually run via run_banding.py.

How bands are suggested
-----------------------
Numeric:
    1. Candidate cut points = ~20 equal-EXPOSURE quantiles, each snapped to a round number
       (leading digits 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5 ... 7.5, 8, 9), so 41,372 -> 40,000
       and 190,000 -> 200,000.
       Low-cardinality whole-number factors (<= 15 values) start with one band per value.
       A whole-number factor with >= 5% of exposure at its minimum (e.g. 0 years trading)
       keeps that value as its own band.
    2. The thinnest band (fewest claims) is merged into its thinner neighbour, repeatedly,
       until every band has >= min_claims and there are <= max_bands.
Categorical:
    Levels with >= min_claims claims and >= min_expo_share of exposure are kept;
    the rest are grouped into OTHER. Levels are ordered by exposure (code 1 = largest).

Conventions
-----------
- Numeric bands are lower-inclusive [a, b): edges [25, 30] give "<25", "25-29", "30+".
- Code 0 is always Missing; codes 1..k follow natural order.
- Relativities are relative to the whole-dataset average (1.00 = average).
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from IPython.display import Markdown, display
except ImportError:  # plain Python: fall back to printing
    Markdown = str
    display = print

MISSING_CODE = 0

# Chart colours: neutral exposure bars; fixed hue per measure (never reassigned)
C_EXPO, C_BC, C_FREQ, C_SEV, C_REF = "#c9c7bf", "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
BLUES = ["#b9d3f3", "#86b3ea", "#5592de", "#2a78d6", "#1d5fb0", "#14467f", "#0c2e55"]


# ---------------------------------------------------------------- inspection

def profile_numeric(df, col, expo="exposure"):
    """Missing rate, cardinality and exposure-weighted percentiles."""
    s, w = df[col], df[expo]
    m = s.notna()
    qs = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    wq = _weighted_quantiles(s[m].to_numpy(float), w[m].to_numpy(float), qs)
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
        g["sev_rel"] = g.sev / (g.cost.sum() / g.claims.sum())
        g["bc_rel"] = g.bc / (g.cost.sum() / g.exposure.sum())
    return g


def plot_one_way(tbl, title="", response="freq"):
    """Exposure bars + a single response line (simple version)."""
    x = [str(i) for i in tbl.index]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x, y=tbl.exposure, name="Exposure", marker_color=C_EXPO), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=tbl[response], name=response, mode="lines+markers",
                             line=dict(color=C_BC, width=2), marker=dict(size=8)), secondary_y=True)
    fig.update_layout(title=title, xaxis_type="category", template="plotly_white")
    fig.update_yaxes(title_text="Exposure", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text=response, secondary_y=True)
    return fig


def stability(df, col, period, expo="exposure", claims="claim_count", cost=None):
    """Relativity by level x period (burning cost if cost given, else frequency)."""
    num = cost or claims
    g = df.groupby([period, col], dropna=False, observed=False)[[expo, num]].sum()
    rate = g[num] / g[expo]
    overall = g.groupby(level=0)[num].sum() / g.groupby(level=0)[expo].sum()
    rel = rate / overall.reindex(g.index.get_level_values(0)).to_numpy()
    return rel.unstack(0)


def cramers_v(df, a, b):
    """Association between two (banded/categorical) factors — flags likely confounding."""
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
    return np.unique(_weighted_quantiles(df.loc[m, col].to_numpy(float), df.loc[m, expo].to_numpy(float), qs))


NICE = np.array([1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 7.5, 8, 9, 10])


def _nice(x):
    """
    Snap to the nearest pricing-friendly round number (leading digits from NICE):
    23.4 -> 25, 37 -> 35, 41,372 -> 40,000, 130,000 -> 125,000, 190,000 -> 200,000.
    """
    if x == 0:
        return 0.0
    sign, x = np.sign(x), abs(x)
    mag = 10 ** np.floor(np.log10(x))
    return float(sign * round(NICE[np.abs(NICE - x / mag).argmin()] * mag, 10))


def _is_whole(s):
    v = s.dropna().to_numpy(float)
    return len(v) > 0 and np.allclose(v, np.round(v))


def candidate_edges(df, col, n=20, expo="exposure", max_levels=15, mass_share=0.05):
    """Round-number candidate cut points. Returns (edges, is_whole_number)."""
    s = df[col]
    whole = _is_whole(s)
    vals = np.sort(s.dropna().unique())
    if len(vals) < 2:
        return [], whole
    if whole and len(vals) <= max_levels:
        return [float(v) for v in vals[1:]], whole          # one band per value
    e = {_nice(v) for v in quantile_edges(df, col, n, expo)}
    if whole:
        e = {float(round(v)) for v in e}
        lo = vals[0]                                        # point mass at the minimum
        if df.loc[s == lo, expo].sum() / df[expo].sum() >= mass_share:
            e.add(float(lo + 1))
    return sorted(v for v in e if vals[0] < v <= vals[-1]), whole


def merge_to_credibility(df, col, edges, min_claims, claims="claim_count", right=False, max_bands=None,
                         cost=None, expo="exposure"):
    """
    Repeatedly merge the thinnest band into a neighbour until every band has >= min_claims
    and (optionally) there are <= max_bands. Missing is ignored.
    With cost given, the neighbour chosen is the one with the closer burning cost, so real
    steps in the curve survive; otherwise the thinner neighbour.
    """
    edges = sorted(edges)
    while edges:
        b = pd.cut(df[col], [-np.inf, *edges, np.inf], right=right)
        cols = [claims] + ([cost, expo] if cost else [])
        g = df.groupby(b, observed=False)[cols].sum()
        c = g[claims].to_numpy()
        i = int(c.argmin())
        if c[i] >= min_claims and (max_bands is None or len(c) <= max_bands):
            break
        if i == 0:
            j = 0
        elif i == len(c) - 1:
            j = i - 1
        elif cost:
            r = np.log(np.clip((g[cost] / g[expo].replace(0, np.nan)).to_numpy(), 1e-9, None))
            dl, dr = abs(r[i] - r[i - 1]), abs(r[i] - r[i + 1])
            if np.isnan(dl) or np.isnan(dr) or np.isclose(dl, dr):
                j = i - 1 if c[i - 1] <= c[i + 1] else i
            else:
                j = i - 1 if dl < dr else i
        else:
            j = i - 1 if c[i - 1] <= c[i + 1] else i
        edges.pop(j)                                        # edge j separates bands j and j+1
    return edges


def suggest_categorical(df, col, min_claims, min_expo_share=0.01, expo="exposure",
                        claims="claim_count", other="OTHER"):
    g = df.groupby(col, observed=True)[[expo, claims]].sum()
    keep = (g[expo] / g[expo].sum() >= min_expo_share) & (g[claims] >= min_claims)
    m = {str(k): (str(k) if keep[k] else other) for k in g.index}
    levels = g[keep].sort_values(expo, ascending=False).index.astype(str).tolist()
    if (~keep).any():
        levels.append(other)
    return {"type": "categorical", "map": m, "levels": levels, "other": other}


# ------------------------------------------------------------- band spec

def _fmt(x):
    """1000000 -> '1,000,000'; 0.25 -> '0.25' (avoids scientific notation)."""
    x = float(x)
    return f"{x:,.0f}" if x.is_integer() else f"{x:,.4f}".rstrip("0").rstrip(".")


def numeric_labels(edges, right=False, integer=True, data_min=None):
    """Human-readable labels matching pd.cut intervals.
    data_min (whole-number factors): label the lowest band from the observed minimum,
    e.g. "0" or "1-3" instead of "<1" / "<4". The band itself still catches anything lower."""
    e = list(edges)
    if not e:
        return ["All"]
    if integer and not right and data_min is not None and data_min < e[0]:
        lo, hi = float(data_min), e[0] - 1
        first = _fmt(lo) if hi == lo else f"{_fmt(lo)}-{_fmt(hi)}"
        mid = [_fmt(a) if b - 1 == a else f"{_fmt(a)}-{_fmt(b - 1)}" for a, b in zip(e, e[1:])]
        return [first] + mid + [f"{_fmt(e[-1])}+"]
    if right:      # (a, b]
        mid = [f"{_fmt(a)}<x<={_fmt(b)}" for a, b in zip(e, e[1:])]
        return [f"<={_fmt(e[0])}"] + mid + [f">{_fmt(e[-1])}"]
    if integer:    # [a, b) on whole numbers -> a to b-1
        mid = [_fmt(a) if b - 1 == a else f"{_fmt(a)}-{_fmt(b - 1)}" for a, b in zip(e, e[1:])]
        return [f"<{_fmt(e[0])}"] + mid + [f"{_fmt(e[-1])}+"]
    mid = [f"{_fmt(a)}<=x<{_fmt(b)}" for a, b in zip(e, e[1:])]
    return [f"<{_fmt(e[0])}"] + mid + [f">={_fmt(e[-1])}"]


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
        other = spec.get("other", "OTHER")
        grouped = s.astype("string").map(spec["map"]).fillna(other)   # unseen levels -> OTHER
        grouped = grouped.where(s.notna(), None)
        lookup = {lvl: i + 1 for i, lvl in enumerate(spec["levels"])}
        return grouped.map(lookup).fillna(MISSING_CODE).astype(int)
    raise ValueError(spec["type"])


def label_table(name, spec):
    """Code -> label lookup, for Emblem level labels and Radar rating tables."""
    if spec["type"] == "numeric":
        labs = numeric_labels(spec["edges"], spec.get("right", False), spec.get("integer", True),
                              spec.get("data_min"))
        lows, highs = [-np.inf, *spec["edges"]], [*spec["edges"], np.inf]
        rows = [(MISSING_CODE, "Missing", None, None)]
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


# ------------------------------------------------------- automated review

def banded_table(df, col, spec, expo="exposure", claims="claim_count", cost=None):
    """One-way table on a band spec, indexed by label, Missing last."""
    codes = apply_band(df[spec.get("source", col)], spec)
    t = one_way(df.assign(_code=codes), "_code", expo, claims, cost)
    t = t.loc[sorted(t.index, key=lambda c: (c == MISSING_CODE, c))]
    labels = label_table(col, spec).set_index("code")["label"]
    t.index = pd.Index(labels.reindex(t.index).to_numpy(), name=col)
    return t


def plot_relativities(t, title, cost=True, min_claims=None):
    """Exposure bars (left axis) + burning cost / frequency / severity relativities (right axis)."""
    x = [str(i) for i in t.index]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=x, y=t.exposure, name="Exposure", marker_color=C_EXPO,
               customdata=np.c_[t.claims, t.expo_share * 100],
               hovertemplate="Exposure %{y:,.0f} (%{customdata[1]:.1f}%)<br>Claims %{customdata[0]:,.0f}<extra></extra>"),
        secondary_y=False,
    )
    lines = [("freq_rel", "Frequency", C_FREQ, 1.5, "dot")]
    if cost:
        lines = [("bc_rel", "Burning cost", C_BC, 3, "solid"),
                 ("freq_rel", "Frequency", C_FREQ, 1.5, "dot"),
                 ("sev_rel", "Severity", C_SEV, 1.5, "dash")]
    for c, name, colour, width, dash in lines:
        fig.add_trace(
            go.Scatter(x=x, y=t[c], name=name, mode="lines+markers",
                       line=dict(color=colour, width=width, dash=dash), marker=dict(size=8 if width > 2 else 6),
                       hovertemplate=f"{name} %{{y:.2f}}<extra></extra>"),
            secondary_y=True,
        )
    if min_claims:   # flag thin bands
        thin = [xi for xi, n in zip(x, t.claims) if n < min_claims]
        if thin:
            fig.add_trace(go.Scatter(x=thin, y=[0] * len(thin), mode="markers", name=f"< {min_claims} claims",
                                     marker=dict(symbol="triangle-up", size=10, color=C_REF),
                                     hoverinfo="skip"), secondary_y=True)
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=1, y1=1,
                  line=dict(color=C_REF, width=1, dash="dot"))
    fig.update_layout(title=dict(text=title, y=0.97), template="plotly_white", hovermode="x unified",
                      height=470, bargap=0.15, margin=dict(t=90, b=20, l=70, r=70),
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
    fig.update_xaxes(type="category", tickangle=-40 if len(x) > 8 else 0, automargin=True)
    fig.update_yaxes(title_text="Exposure", secondary_y=False, showgrid=False, tickformat=",.0f")
    fig.update_yaxes(title_text="Relativity (1 = average)", secondary_y=True, rangemode="tozero",
                     tickmode="auto", nticks=6, tickformat=".1f", showgrid=True)
    return fig


def plot_stability(rel, title):
    """One line per period (light = oldest, dark = newest)."""
    periods = list(rel.columns)
    idx = np.linspace(0, len(BLUES) - 1, len(periods)).round().astype(int) if len(periods) > 1 else [3]
    fig = go.Figure()
    for p, k in zip(periods, idx):
        name = str(_tidy(p)) if isinstance(p, (int, float, np.number)) else str(p)   # 2021.0 -> 2021
        fig.add_trace(go.Scatter(x=[str(i) for i in rel.index], y=rel[p], name=name, mode="lines+markers",
                                 line=dict(color=BLUES[k], width=2), marker=dict(size=7),
                                 hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=C_REF, width=1, dash="dot"))
    fig.update_layout(title=dict(text=title, y=0.97), template="plotly_white", hovermode="x unified",
                      height=400, yaxis_title="Relativity (1 = average)", yaxis_rangemode="tozero",
                      margin=dict(t=90, b=20, l=70, r=70),
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
    fig.update_xaxes(type="category", tickangle=-40 if len(rel) > 8 else 0, automargin=True)
    return fig


# ------------------------------------------------------- raw (granular) view

def _smooth(num, den, k):
    """Credibility-weighted smoother: triangular kernel over neighbouring bins applied to
    numerator and denominator separately, so thin bins pull the line less than thick ones."""
    w = np.bartlett(2 * k + 3)[1:-1]
    n = np.convolve(np.asarray(num, float), w, mode="same")
    d = np.convolve(np.asarray(den, float), w, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        return n / d


def raw_view_table(df, col, expo="exposure", claims="claim_count", cost=None, categorical=False,
                   raw_bins=100, raw_max_levels=100, max_cat_levels=150):
    """
    As-close-to-raw-as-possible one-way: every value for whole-number factors with up to
    raw_max_levels values, otherwise raw_bins equal-exposure bins (unrounded, unmerged);
    every level for categoricals (largest max_cat_levels). Adds smoothed relativities.
    Returns (table, kind) with kind in {"values", "bins", "categorical"}.
    """
    tot_e, tot_n = df[expo].sum(), df[claims].sum()
    tot_c = df[cost].sum() if cost else None
    s = df[col]
    agg = {"exposure": (expo, "sum"), "claims": (claims, "sum")}
    if cost:
        agg["cost"] = (cost, "sum")
    if categorical:
        g = df.assign(_k=s.astype("string")).groupby("_k", observed=True).agg(**agg)
        g = g.sort_values("exposure", ascending=False).head(max_cat_levels)
        g["x"] = g.index.astype(str)
        g["label"] = g["x"]
        kind = "categorical"
    else:
        m = s.notna()
        d = df.loc[m]
        v = s[m]
        if v.nunique() <= raw_max_levels:
            key, kind = v.to_numpy(), "values"
        else:
            edges = quantile_edges(d, col, raw_bins, expo)
            key, kind = pd.cut(v, [-np.inf, *edges, np.inf], right=False, labels=False).to_numpy(), "bins"
        d = d.assign(_k=key, _xw=v * d[expo])
        g = d.groupby("_k").agg(**agg, xw=("_xw", "sum"), lo=(col, "min"), hi=(col, "max"))
        g["x"] = g.xw / g.exposure                       # exposure-weighted mean value of the bin
        g = g.sort_values("x")
        g["label"] = [_fmt(lo) if lo == hi else f"{_fmt(lo)} – {_fmt(hi)}" for lo, hi in zip(g.lo, g.hi)]
    g["freq_rel"] = (g.claims / g.exposure) / (tot_n / tot_e)
    if cost:
        g["bc_rel"] = (g.cost / g.exposure) / (tot_c / tot_e)
        with np.errstate(divide="ignore", invalid="ignore"):
            g["sev_rel"] = (g.cost / g.claims) / (tot_c / tot_n)
    if kind != "categorical" and len(g) >= 5:
        k = max(1, round(len(g) / 20))
        g["freq_smooth"] = _smooth(g.claims, g.exposure, k) / (tot_n / tot_e)
        if cost:
            g["bc_smooth"] = _smooth(g.cost, g.exposure, k) / (tot_c / tot_e)
            g["sev_smooth"] = _smooth(g.cost, g.claims, k) / (tot_c / tot_n)
    return g, kind


def top_values(df, col, expo="exposure", n=8):
    """Most common values by exposure — shows spikes at defaults or round numbers."""
    g = df.groupby(col, observed=True)[expo].sum().sort_values(ascending=False).head(n)
    return pd.DataFrame({"exposure": g.to_numpy(), "expo_share": (g / df[expo].sum()).to_numpy()},
                        index=g.index.map(_fmt_any))


def _fmt_any(v):
    try:
        return _fmt(v)
    except (TypeError, ValueError):
        return str(v)


def plot_raw(t, kind, title, values=None, weights=None, cost=True, min_claims=100):
    """
    Top: observed relativity per value/bin (dot size = claims) with a smoothed line.
    Bottom: where the exposure sits (exposure per value, or an exposure histogram).
    Two panels sharing the x axis — no second y axis.
    """
    main, smooth = ("bc_rel", "bc_smooth") if cost else ("freq_rel", "freq_smooth")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    y = t[main].to_numpy(float)
    finite = y[np.isfinite(y)]
    ref = [3.0, np.nanpercentile(finite, 95) * 1.15 if len(finite) else 3.0]
    if smooth in t:
        ref.append(np.nanmax(t[smooth]) * 1.4)
    ymax = float(max(ref))
    clipped = y > ymax
    size = np.clip(4 + 2.2 * np.sqrt(t.claims.to_numpy(float)), 5, 22)
    x = t["x"].tolist()
    fig.add_trace(go.Scatter(
        x=x, y=np.where(clipped, ymax * 0.98, y), mode="markers", name="Observed (dot size = claims)",
        marker=dict(size=size, color=C_BC, opacity=0.45, symbol=np.where(clipped, "triangle-up", "circle"),
                    line=dict(width=0)),
        customdata=np.c_[t["label"], t.claims, t.exposure, y],
        hovertemplate="%{customdata[0]}<br>Relativity %{customdata[3]:.2f}<br>Claims %{customdata[1]:,.0f}"
                      "<br>Exposure %{customdata[2]:,.0f}<extra></extra>"), row=1, col=1)
    if smooth in t:
        lines = [(smooth, "Burning cost (smoothed)" if cost else "Frequency (smoothed)", C_BC, 3, "solid")]
        if cost:
            lines += [("freq_smooth", "Frequency (smoothed)", C_FREQ, 1.5, "dot"),
                      ("sev_smooth", "Severity (smoothed)", C_SEV, 1.5, "dash")]
        for c, name, colour, width, dash in lines:
            fig.add_trace(go.Scatter(x=x, y=t[c], mode="lines", name=name,
                                     line=dict(color=colour, width=width, dash=dash),
                                     hovertemplate=f"{name.split(' (')[0]} %{{y:.2f}}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=1, line=dict(color=C_REF, width=1, dash="dot"), row=1, col=1)

    logx, zero_note = False, ""
    if kind == "bins" and values is not None:
        v = np.asarray(values, float)
        w = np.asarray(weights, float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        pos = v > 0
        zero_share = w[v == 0].sum() / w.sum()
        if pos.sum() > 10 and (v >= 0).all():
            p01, p99 = _weighted_quantiles(v[pos], w[pos], [0.01, 0.99])
            logx = p99 / max(p01, 1e-12) > 50
        if logx:
            min_pos = v[pos].min()
            edges = np.logspace(np.log10(min_pos), np.log10(hi), 61)
            if zero_share > 0:                        # zeros can't sit on a log axis: plot them just left
                zx = min_pos / 3
                fig.data[0].x = tuple(zx if xi == 0 else xi for xi in fig.data[0].x)
                for tr in fig.data[1:]:
                    tr.x = tuple(zx if xi == 0 else xi for xi in tr.x)
                fig.add_trace(go.Scatter(x=[zx, zx], y=[0, w[v == 0].sum()], mode="lines", showlegend=False,
                                         line=dict(color=C_REF, width=6),
                                         hovertemplate=f"Zero: exposure %{{y:,.0f}}<extra></extra>"), row=2, col=1)
                fig.add_annotation(x=np.log10(zx), y=0, xref="x2", yref="y2 domain", text="0", showarrow=False,
                                   yshift=-14, font=dict(color=C_REF, size=11))
                zero_note = f", zeros ({zero_share:.0%} of exposure) shown at the far left"
            v, w = v[pos], w[pos]
        else:
            hi_view = _weighted_quantiles(v, w, [0.995])[0]
            edges = np.linspace(lo, hi_view, 61)
        h, _ = np.histogram(v, bins=edges, weights=w)
        fig.add_trace(go.Scatter(x=np.r_[edges[0], edges[1:]], y=np.r_[h[0], h], mode="lines",
                                 line=dict(color=C_EXPO, width=1, shape="vh"), fill="tozeroy",
                                 fillcolor="rgba(201,199,191,0.6)", name="Exposure distribution",
                                 hovertemplate="Exposure %{y:,.0f}<extra></extra>"), row=2, col=1)
        if not logx:
            fig.update_xaxes(range=[lo, hi_view])
    else:
        fig.add_trace(go.Bar(x=x, y=t.exposure, marker_color=C_EXPO, name="Exposure",
                             hovertemplate="Exposure %{y:,.0f}<extra></extra>"), row=2, col=1)
    if logx:
        fig.update_xaxes(type="log")
    if kind == "categorical":
        fig.update_xaxes(type="category", tickangle=-40 if len(x) > 8 else 0)
    fig.update_yaxes(title_text="Relativity", range=[0, ymax], row=1, col=1, tickformat=".1f")
    fig.update_yaxes(title_text="Exposure", row=2, col=1, tickformat=",.0f", rangemode="tozero")
    note = {"values": "every value", "bins": f"{len(t)} equal-exposure bins, unrounded",
            "categorical": f"every level ({len(t)})"}[kind]
    if kind == "bins" and logx:
        note += ", log scale" + zero_note
    elif kind == "bins":
        note += ", x axis cut at the 99.5th percentile"
    fig.update_layout(title=dict(text=f"{title} ({note})", y=0.97), template="plotly_white",
                      hovermode="closest", height=560, margin=dict(t=90, b=20, l=70, r=40),
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"), bargap=0.1)
    fig.update_xaxes(automargin=True)
    return fig


# ------------------------------------------------------- peril comparison

PERIL_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def fine_spec_for(df, col, categorical=False, expo="exposure", n_candidates=20, max_fine_levels=40):
    """The fine-view band spec (round-number equal-exposure grid / top categorical levels)."""
    s = df[col]
    if not categorical:
        cand, whole = candidate_edges(df, col, n_candidates, expo)
        return {"type": "numeric", "edges": cand, "right": False, "integer": whole,
                "data_min": _tidy(s.min()) if whole else None}
    g = df.groupby(col, observed=True)[expo].sum().sort_values(ascending=False)
    top = g.index[:max_fine_levels].astype(str).tolist()
    rest = "(other levels)" if len(g) > max_fine_levels else None
    return {"type": "categorical", "map": {str(k): (str(k) if str(k) in top else rest) for k in g.index},
            "levels": top + ([rest] if rest else []), "other": rest or "OTHER"}


def peril_tables(df, col, spec, perils, expo="exposure"):
    """perils: {name: (count_col, cost_col)}. Returns (bc relativity, claims, cost) tables: band x peril."""
    codes = apply_band(df[spec.get("source", col)], spec)
    d = df.assign(_code=codes)
    cols = [expo] + [c for p in perils.values() for c in p]
    g = d.groupby("_code")[cols].sum()
    g = g.loc[sorted(g.index, key=lambda c: (c == MISSING_CODE, c))]
    labels = label_table(col, spec).set_index("code")["label"]
    idx = pd.Index(labels.reindex(g.index).to_numpy(), name=col)
    rel, n, c = {}, {}, {}
    for p, (cc, kc) in perils.items():
        rel[p] = (g[kc] / g[expo]) / (d[kc].sum() / d[expo].sum())
        n[p], c[p] = g[cc], g[kc]
    out = [pd.DataFrame(x) for x in (rel, n, c)]
    for t in out:
        t.index = idx
    return out


def plot_peril_lines(rel, n, colours, title, min_claims=100):
    """Burning cost relativity by peril. Hollow markers = fewer than that peril's min claims
    (min_claims can be a number or {peril: number})."""
    x = [str(i) for i in rel.index]
    fig = go.Figure()
    for p in rel.columns:
        mc = min_claims.get(p, min(min_claims.values())) if isinstance(min_claims, dict) else min_claims
        thin = n[p].to_numpy() < mc
        fig.add_trace(go.Scatter(
            x=x, y=rel[p], name=p, mode="lines+markers", line=dict(color=colours[p], width=2),
            marker=dict(size=8, color=colours[p], symbol=np.where(thin, "circle-open", "circle")),
            customdata=n[p].to_numpy(),
            hovertemplate=f"{p}: %{{y:.2f}} (%{{customdata:,.0f}} claims)<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=C_REF, width=1, dash="dot"))
    fig.update_layout(title=dict(text=title, y=0.97), template="plotly_white", hovermode="x unified",
                      height=430, yaxis_title="Burning cost relativity", yaxis_rangemode="tozero",
                      margin=dict(t=90, b=20, l=70, r=40), legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
    fig.update_xaxes(type="category", tickangle=-40 if len(x) > 8 else 0, automargin=True)
    return fig


def plot_peril_mix(c, colours, title):
    """Share of claim cost by peril within each band (100% stacked)."""
    share = c.div(c.sum(axis=1).replace(0, np.nan), axis=0)
    x = [str(i) for i in share.index]
    fig = go.Figure()
    for p in share.columns:
        fig.add_trace(go.Bar(x=x, y=share[p], name=p, marker=dict(color=colours[p], line=dict(color="white", width=1)),
                             hovertemplate=f"{p}: %{{y:.0%}}<extra></extra>"))
    fig.update_layout(barmode="stack", title=dict(text=title, y=0.97), template="plotly_white",
                      hovermode="x unified", height=400, yaxis=dict(title="Share of claim cost", tickformat=".0%",
                                                                     range=[0, 1]),
                      margin=dict(t=90, b=20, l=70, r=40),
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", traceorder="normal"),
                      bargap=0.15)
    fig.update_xaxes(type="category", tickangle=-40 if len(x) > 8 else 0, automargin=True)
    return fig


def peril_overview(df, factors, perils, expo="exposure", categorical=(), n_candidates=20,
                   max_fine_levels=40, min_claims=100, report=None, show=False):
    """
    For each factor, on its fine grid: burning cost relativity by peril and the peril mix of
    claim cost. More than 8 perils: the smallest are grouped as "Other perils" in these charts.
    """
    totals = {p: df[k].sum() for p, (_, k) in perils.items()}
    order = sorted(perils, key=lambda p: -totals[p])
    d = df
    if len(order) > 8:
        keep, rest = order[:7], order[7:]
        d = df.assign(_oth_n=df[[perils[p][0] for p in rest]].sum(axis=1),
                      _oth_c=df[[perils[p][1] for p in rest]].sum(axis=1))
        perils = {**{p: perils[p] for p in keep}, "Other perils": ("_oth_n", "_oth_c")}
        order = keep + ["Other perils"]
    perils = {p: perils[p] for p in order}
    colours = {p: PERIL_COLOURS[i] for i, p in enumerate(order)}
    for col in factors:
        print(f"peril overview: {col} ...")
        spec = fine_spec_for(d, col, col in categorical, expo, n_candidates, max_fine_levels)
        rel, n, c = peril_tables(d, col, spec, perils, expo)
        f1 = plot_peril_lines(rel, n, colours, f"{col} — burning cost relativity by peril (fine view)", min_claims)
        f2 = plot_peril_mix(c, colours, f"{col} — peril mix of claim cost (fine view)")
        if show:
            display(Markdown(f"---\n## {col}"))
            f1.show()
            f2.show()
        if report is not None:
            report.add_block_section(col, [("Burning cost relativity by peril", f1,
                                             "Hollow markers have fewer claims than that peril's minimum per band."),
                                            ("Peril mix", f2, "Share of claim cost by peril within each band.")])


def review_variable(df, col, expo="exposure", claims="claim_count", cost=None, period=None,
                    spec=None, categorical=False, min_claims=100, max_bands=15, n_candidates=20,
                    min_expo_share=0.01, max_fine_levels=40, show=True, report=None,
                    show_raw=True, raw_bins=100, raw_max_levels=100, override_hint="OVERRIDES"):
    """
    Profile one factor, chart it on a fine grid, suggest bands (or use `spec` if given),
    chart the bands and their stability by period.
    show=True displays in a Jupyter notebook; report=HtmlReport(...) adds it to an HTML report.
    Returns (spec, summary_dict).
    """
    s = df[col]
    is_num = not categorical and pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    rel_col = "bc_rel" if cost else "freq_rel"
    overridden = spec is not None

    fine_spec = fine_spec_for(df, col, not is_num, expo, n_candidates, max_fine_levels)
    if spec is None and is_num:
        edges = merge_to_credibility(df, col, fine_spec["edges"], min_claims, claims, max_bands=max_bands,
                                     cost=cost, expo=expo)
        spec = {**fine_spec, "edges": [_tidy(e) for e in edges]}
    elif spec is None:
        spec = suggest_categorical(df, col, min_claims, min_expo_share, expo, claims)

    fine = banded_table(df, col, fine_spec, expo, claims, cost)
    sugg = banded_table(df, col, spec, expo, claims, cost)
    cred = sugg[sugg.claims >= min_claims]

    info = {
        "variable": col,
        "type": "numeric" if is_num else "categorical",
        "bands_from": "override" if overridden else "suggested",
        "distinct": s.nunique(),
        "missing_expo_%": round(100 * df.loc[s.isna(), expo].sum() / df[expo].sum(), 1),
        "bands": len(sugg),
        "thin_bands": int((sugg.claims < min_claims).sum()),
        "min_band_claims": int(sugg.claims.min()),
        "rel_min": round(cred[rel_col].min(), 2) if len(cred) else np.nan,
        "rel_max": round(cred[rel_col].max(), 2) if len(cred) else np.nan,
    }
    info["rel_spread"] = round(info["rel_max"] / info["rel_min"], 2) if len(cred) else np.nan

    if not (show or report is not None):
        return spec, info

    measure = "burning cost" if cost else "frequency"
    word = "Your bands (override)" if overridden else "Suggested bands"
    cols = ["exposure", "expo_share", "claims", "freq_rel"] + (["sev_rel", "bc_rel"] if cost else [])
    fig_raw, tops = None, None
    if show_raw:
        raw, kind = raw_view_table(df, col, expo, claims, cost, not is_num, raw_bins, raw_max_levels)
        m = s.notna()
        fig_raw = plot_raw(raw, kind, f"{col} — raw view", values=s[m].to_numpy(float) if kind == "bins" else None,
                           weights=df.loc[m, expo].to_numpy(float) if kind == "bins" else None,
                           cost=bool(cost), min_claims=min_claims)
        if kind == "bins":
            tops = top_values(df, col, expo)
    fig_fine = plot_relativities(fine, f"{col} — fine view ({len(fine)} levels)", bool(cost), min_claims)
    fig_sugg = plot_relativities(sugg, f"{col} — {word.lower()}", bool(cost), min_claims)
    fig_stab = None
    if period:
        codes = apply_band(df[spec.get("source", col)], spec)
        rel = stability(df.assign(_code=codes), "_code", period, expo, claims, cost)
        rel = rel.loc[sorted(rel.index, key=lambda c: (c == MISSING_CODE, c))]
        labels = label_table(col, spec).set_index("code")["label"]
        rel.index = labels.reindex(rel.index).to_numpy()
        fig_stab = plot_stability(rel, f"{col} — {word.lower()} by {period} ({measure} relativity)")
    spec_text = (f'"{col}": {{"edges": {spec["edges"]}}},' if is_num else
                 f'"{col}": {{"levels": {json.dumps(spec["levels"])}}},')     # paste into OVERRIDES
    profile = profile_numeric(df, col, expo).to_frame().T if is_num else None

    if show:
        display(Markdown(f"---\n## {col}"))
        if profile is not None:
            display(profile.round(2))
        display(pd.DataFrame([info]).set_index("variable"))
        if fig_raw is not None:
            fig_raw.show()
        fig_fine.show()
        display(Markdown(f"**{word}** (min {min_claims} claims, max {max_bands} bands): `{spec_text}`"))
        display(sugg[cols].round(3))
        fig_sugg.show()
        if fig_stab is not None:
            fig_stab.show()

    if report is not None:
        report.add_factor(col, info, profile, fig_fine, word, spec_text, sugg[cols], fig_sugg, fig_stab,
                          min_claims, max_bands, fig_raw=fig_raw, top_values=tops, override_hint=override_hint)
    return spec, info


def _tidy(x):
    x = float(x)
    return int(x) if x.is_integer() else x


def review_all(df, factors, expo="exposure", claims="claim_count", cost=None, period=None,
               categorical=(), overrides=None, show=True, report=None, label="", **kw):
    """
    Loop review_variable over every factor. Returns (BANDS, summary).
    categorical: factors to treat as categorical even if stored as numbers (e.g. trade codes).
    overrides:   {factor: spec} to use instead of the suggestion.
    report:      an HtmlReport to add every factor to (call report.save(...) afterwards).
    """
    overrides = overrides or {}
    bands, rows = {}, []
    for col in factors:
        print(f"reviewing {label + ': ' if label else ''}{col} ...")
        spec, info = review_variable(df, col, expo, claims, cost, period, spec=overrides.get(col),
                                     categorical=col in categorical, show=show, report=report, **kw)
        bands[col] = spec
        rows.append(info)
    summary = pd.DataFrame(rows).set_index("variable").sort_values("rel_spread", ascending=False)
    if show:
        display(Markdown("---\n## Summary (sorted by relativity spread across credible bands)"))
        display(summary)
    if report is not None:
        report.summary = summary
    return bands, summary


# ---------------------------------------------------------------- data loading

def load_data(path, columns=None, encoding=None):
    """
    Read a SAS dataset (.sas7bdat / .xpt), .csv or .parquet. Column names are matched
    case-insensitively (SAS names are often upper case) and returned exactly as requested.
    Uses pyreadstat for SAS files if installed (faster, reads only the columns needed).
    """
    path = str(path)
    ext = path.lower().rsplit(".", 1)[-1]
    want = list(dict.fromkeys(columns)) if columns else None

    def _match(available):
        lookup = {c.lower(): c for c in available}
        missing = [c for c in want if c.lower() not in lookup]
        if missing:
            raise KeyError(f"not found in {path}: {missing}\navailable: {sorted(available)}")
        return {lookup[c.lower()]: c for c in want}       # actual name -> requested name

    if ext in ("sas7bdat", "xpt"):
        try:
            import pyreadstat
            reader = pyreadstat.read_sas7bdat if ext == "sas7bdat" else pyreadstat.read_xport
            kw = {"encoding": encoding} if encoding else {}
            rename = None
            if want:
                _, meta = reader(path, metadataonly=True, **kw)
                rename = _match(meta.column_names)
                kw["usecols"] = list(rename)
            df, _ = reader(path, **kw)
        except ImportError:
            df = pd.read_sas(path, format="sas7bdat" if ext == "sas7bdat" else "xport",
                             encoding=encoding or "latin-1")
            rename = _match(df.columns) if want else None
    elif ext == "csv":
        df = pd.read_csv(path, low_memory=False)
        rename = _match(df.columns) if want else None
    elif ext == "parquet":
        df = pd.read_parquet(path)
        rename = _match(df.columns) if want else None
    else:
        raise ValueError(f"unsupported file type: {path}")
    if rename:
        df = df[list(rename)].rename(columns=rename)
    return df


def prepare(df, expo, claims, cost=None):
    """
    Basic cleaning for burning cost / Tweedie data. claims and cost can be a column name or a
    list (one per peril). Blank claims -> 0, zero/blank exposure dropped, negative cost -> 0.
    Returns (df, list_of_notes).
    """
    counts = [claims] if isinstance(claims, str) else list(claims)
    costs = [] if cost is None else ([cost] if isinstance(cost, str) else list(cost))
    notes, blanks = [], {}
    for c in counts + costs:
        n = int(df[c].isna().sum())
        if n:
            df[c] = df[c].fillna(0)
            blanks[c] = n
    if len(blanks) > 2:
        notes.append(f"Blank claim counts/costs set to 0 (no claims) in {len(blanks)} columns, "
                     f"{min(blanks.values()):,}–{max(blanks.values()):,} rows each")
    else:
        notes += [f"{n:,} rows with blank {c} set to 0 (no claims)" for c, n in blanks.items()]
    bad = df[expo].isna() | (df[expo] <= 0)
    if bad.any():
        notes.append(f"{int(bad.sum()):,} rows with zero, negative or blank {expo} removed")
        df = df[~bad].copy()
    for c in costs:
        neg = df[c] < 0
        if neg.any():
            notes.append(f"{int(neg.sum()):,} rows with negative {c} floored at 0 (total {df.loc[neg, c].sum():,.0f})")
            df.loc[neg, c] = 0
    for cc, kc in zip(counts, costs):
        odd = (df[cc] == 0) & (df[kc] > 0)
        if odd.any():
            notes.append(f"{int(odd.sum()):,} rows with {kc} > 0 but {cc} = 0 — check (e.g. reopened/IBNR claims)")
    if not notes:
        notes.append("no issues found")
    for n in notes:
        print("data check:", n)
    return df, notes


# ---------------------------------------------------------------- HTML report

_CSS = """
:root { --ink:#1f1f1d; --ink2:#55534e; --muted:#8a8984; --line:#e4e2dc; --bg:#ffffff; --panel:#f7f6f3;
        --accent:#2a78d6; --thin:#fdf3dc; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width:1150px; margin:0 auto; padding:24px 20px 80px; }
h1 { font-size:24px; margin:0 0 4px; }
h2 { font-size:20px; margin:0 0 10px; }
h3 { font-size:15px; margin:22px 0 8px; color:var(--ink2); }
.sub { color:var(--ink2); margin:0 0 20px; }
section { border-top:1px solid var(--line); padding-top:24px; margin-top:32px; }
.meta { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:8px 24px;
        background:var(--panel); padding:14px 16px; border-radius:6px; }
.meta div span { display:block; color:var(--muted); font-size:12px; }
.stats { display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 14px; }
.stat { background:var(--panel); border-radius:6px; padding:8px 12px; min-width:110px; }
.stat span { display:block; color:var(--muted); font-size:12px; }
.stat b { font-size:16px; font-weight:600; }
.tag { display:inline-block; font-size:12px; padding:1px 8px; border-radius:10px; background:var(--panel);
       color:var(--ink2); margin-left:8px; vertical-align:middle; font-weight:normal; }
.tbl { overflow-x:auto; }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; margin:4px 0 8px; }
th, td { padding:5px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th { color:var(--ink2); font-weight:600; font-size:12px; border-bottom:1px solid var(--muted); }
th:first-child, td:first-child { text-align:left; }
tr.thin td { background:var(--thin); }
code, pre { font-family:Consolas, "SFMono-Regular", Menlo, monospace; font-size:13px; }
pre { background:var(--panel); padding:10px 12px; border-radius:6px; white-space:pre-wrap; word-break:break-all; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
.top { font-size:12px; float:right; }
.note { color:var(--ink2); font-size:13px; }
ul.checks { margin:6px 0; padding-left:20px; }
details.blk { margin:14px 0; border:1px solid var(--line); border-radius:6px; padding:0 12px; }
details.blk > summary { cursor:pointer; padding:9px 0; font-weight:600; color:var(--ink2); }
details.blk[open] > summary { border-bottom:1px solid var(--line); margin-bottom:8px; }
.heat td.h { color:var(--ink); }
.cols2 { display:flex; flex-wrap:wrap; gap:24px; }
"""

_JS = """
document.addEventListener('toggle', function (e) {
  if (e.target.tagName === 'DETAILS' && e.target.open && window.Plotly) {
    e.target.querySelectorAll('.js-plotly-plot').forEach(function (p) { Plotly.Plots.resize(p); });
  }
}, true);
"""

_BAND_FMT = {"exposure": ",.0f", "expo_share": ".1%", "claims": ",.0f",
             "freq_rel": ".2f", "sev_rel": ".2f", "bc_rel": ".2f"}
_BAND_HDR = {"exposure": "Exposure", "expo_share": "Exposure %", "claims": "Claims",
             "freq_rel": "Frequency rel.", "sev_rel": "Severity rel.", "bc_rel": "Burning cost rel."}


def _esc(x):
    import html
    return html.escape(str(x))


def _cell(v, f=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if f:
        return format(v, f)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}".rstrip("0").rstrip(".") if v % 1 else f"{v:,.0f}"
    return _esc(v)


def _table(df, fmt=None, headers=None, row_class=None, index_name=None, links=None):
    fmt, headers = fmt or {}, headers or {}
    h = [f"<th>{_esc(index_name or df.index.name or '')}</th>"]
    h += [f"<th>{_esc(headers.get(c, c))}</th>" for c in df.columns]
    rows = []
    for i, (idx, r) in enumerate(df.iterrows()):
        cls = f' class="{row_class[i]}"' if row_class is not None and row_class[i] else ""
        first = f'<a href="#{links[idx]}">{_esc(idx)}</a>' if links else _esc(idx)
        cells = [f"<td>{first}</td>"] + [f"<td>{_cell(r[c], fmt.get(c))}</td>" for c in df.columns]
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f'<div class="tbl"><table><thead><tr>{"".join(h)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def heat_table(df, index_name="", fmt=".2f", links=None):
    """Table with cells shaded light-to-dark blue by value (log scale; NaN blank)."""
    v = np.log(df.astype(float).clip(lower=1))
    hi = np.nanmax(v.to_numpy()) if np.isfinite(v.to_numpy()).any() else 1
    def shade(x):
        if not np.isfinite(x):
            return ""
        t = 0 if hi <= 0 else min(1, x / hi)
        r, g, b = [int(round(a + (z - a) * t)) for a, z in zip((247, 246, 243), (85, 146, 222))]
        return f' style="background:rgb({r},{g},{b})"'
    h = f"<th>{_esc(index_name)}</th>" + "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    rows = []
    for idx, r in df.iterrows():
        first = f'<a href="#{links[idx]}">{_esc(idx)}</a>' if links else _esc(idx)
        cells = "".join(f'<td class="h"{shade(v.loc[idx, c])}>{_cell(r[c], fmt)}</td>' for c in df.columns)
        rows.append(f"<tr><td>{first}</td>{cells}</tr>")
    return f'<div class="tbl heat"><table><thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _slug(s):
    import re
    return "f-" + re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


class HtmlReport:
    """Collects review output and writes one self-contained HTML file (works offline)."""

    def __init__(self, title="Banding review", subtitle=""):
        self.title, self.subtitle = title, subtitle
        self.sections, self.summary, self.summary_html = [], None, None
        self.meta, self.checks = {}, []

    @staticmethod
    def _blk(title, body, open_=True):
        return f'<details class="blk"{" open" if open_ else ""}><summary>{_esc(title)}</summary>{body}</details>'

    def add_block_section(self, col, blocks, intro=""):
        """Generic factor section: blocks = [(title, fig_or_html, note), ...]."""
        parts = []
        for title, obj, note in blocks:
            body = (f'<p class="note">{_esc(note)}</p>' if note else "") + (
                obj if isinstance(obj, str) else self._fig(obj))
            parts.append(self._blk(title, body))
        self.sections.append(f"""
<section id="{_slug(col)}">
  <a class="top" href="#top">back to top</a>
  <h2>{_esc(col)}</h2>{intro}
  {''.join(parts)}
</section>""")

    @staticmethod
    def _fig(fig):
        if fig is None:
            return ""
        return fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%",
                           config={"displaylogo": False, "responsive": True})

    def add_factor(self, col, info, profile, fig_fine, word, spec_text, table, fig_sugg, fig_stab,
                   min_claims, max_bands, fig_raw=None, top_values=None, override_hint="OVERRIDES"):
        sid = _slug(col)
        stats = [("Type", info["type"]), ("Distinct values", f"{info['distinct']:,}"),
                 ("Missing exposure", f"{info['missing_expo_%']}%"), ("Bands", info["bands"]),
                 ("Bands under min claims", info["thin_bands"]),
                 ("Fewest claims in a band", f"{info['min_band_claims']:,}"),
                 ("Relativity range", f"{_cell(info['rel_min'], '.2f')} – {_cell(info['rel_max'], '.2f')}")]
        stat_html = "".join(f'<div class="stat"><span>{_esc(k)}</span><b>{_esc(v)}</b></div>' for k, v in stats)
        prof_html = ""
        if profile is not None:
            p = profile.copy()
            p.columns = [c.replace("wq", "p") if c.startswith("wq") else c for c in p.columns]
            hdr = {"rows": "Rows", "missing_rows": "Missing rows", "missing_expo_share": "Missing exposure",
                   "distinct": "Distinct", "p00": "Min", "p100": "Max"}
            prof_html = ("<h3>Profile (exposure-weighted percentiles)</h3>"
                         + _table(p, fmt={"missing_expo_share": ".1%"}, headers=hdr, index_name="Factor"))
        if top_values is not None:
            prof_html += ("<h3>Most common values</h3><p class='note'>Large shares at one value often mean a "
                          "default or a rounded input.</p>"
                          + _table(top_values, {"exposure": ",.0f", "expo_share": ".1%"},
                                   {"exposure": "Exposure", "expo_share": "Exposure %"}, index_name="Value"))
        thin = ["thin" if n < min_claims else "" for n in table.claims]
        tag = "override" if info["bands_from"] == "override" else "suggested"
        raw_html = ""
        if fig_raw is not None:
            raw_html = self._blk("Raw view",
                '<p class="note">Each dot is one value (or one small bin); bigger dots have more claims. '
                'The line is a smoothed trend that gives more weight to dots with more claims. '
                'Triangles at the top are cut off: hover for the value.</p>' + self._fig(fig_raw))
        band_body = (f'<p class="note">Minimum {min_claims:,} claims per band, at most {max_bands} bands. '
                     f'Highlighted rows are below the minimum. Copy the line below into '
                     f'{_esc(override_hint)} to change it.</p>'
                     f'<pre>{_esc(spec_text)}</pre>'
                     + _table(table, _BAND_FMT, _BAND_HDR, thin, index_name="Band") + self._fig(fig_sugg))
        stab_html = self._blk("Stability by period", self._fig(fig_stab)) if fig_stab is not None else ""
        self.sections.append(f"""
<section id="{sid}">
  <a class="top" href="#top">back to top</a>
  <h2>{_esc(col)}<span class="tag">{tag} bands</span></h2>
  <div class="stats">{stat_html}</div>
  {prof_html}
  {raw_html}
  {self._blk("Fine view", self._fig(fig_fine))}
  {self._blk(word, band_body)}
  {stab_html}
</section>""")

    def save(self, path, plotlyjs="inline"):
        """plotlyjs='inline' embeds plotly.js (~4.5MB, works offline); 'cdn' keeps the file small."""
        import datetime
        from plotly.offline import get_plotlyjs, get_plotlyjs_version
        js = (f"<script>{get_plotlyjs()}</script>" if plotlyjs == "inline"
              else f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>')
        meta = {"Generated": datetime.datetime.now().strftime("%d %b %Y %H:%M"), **self.meta}
        meta_html = "".join(f"<div><span>{_esc(k)}</span>{_esc(v)}</div>" for k, v in meta.items())
        checks = "".join(f"<li>{_esc(c)}</li>" for c in self.checks)
        summ = self.summary_html or ""
        if self.summary is not None and not self.summary_html:
            s = self.summary.drop(columns=["rel_min", "rel_max"], errors="ignore")
            hdr = {"type": "Type", "bands_from": "Bands", "distinct": "Distinct", "missing_expo_%": "Missing exp. %",
                   "bands": "No. bands", "thin_bands": "Under min claims", "min_band_claims": "Fewest claims",
                   "rel_spread": "Relativity spread"}
            summ = ("<h2>Summary</h2><p class='note'>Sorted by relativity spread (highest ÷ lowest relativity "
                    "across bands that meet the minimum claims) — a rough one-way guide to which factors "
                    "move the most. Click a factor to jump to it.</p>"
                    + _table(s, {"rel_spread": ".2f", "missing_expo_%": ".1f"}, hdr, index_name="Factor",
                             links={i: _slug(i) for i in s.index}))
        html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(self.title)}</title><style>{_CSS}</style>{js}</head>
<body><main id="top">
<h1>{_esc(self.title)}</h1><p class="sub">{_esc(self.subtitle)}</p>
<div class="meta">{meta_html}</div>
{('<h3>Data checks</h3><ul class="checks">' + checks + '</ul>') if checks else ''}
<section style="border:0;margin-top:8px">{summ}</section>
{''.join(self.sections)}
</main><script>{_JS}</script></body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        return path


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
                print(f"WARNING {name}{suffix}: codes {list(thin.index)} below {min_claims} claims")
    return report
