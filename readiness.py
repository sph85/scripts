"""
Data readiness and feature engineering (step 1, before screening and banding).

    engineer(df, cfg)            apply confirmed mappings, junk rules and "not applicable" recodes
    detect_structural(...)       propose "not applicable" recodes from where missing / zero values sit
    factor_checks(...)           per-factor checks: missing pattern, junk, spikes, product overlap, 255 levels
    product_summary / cover_check
    build_report(...)            HTML readiness report
    write_spec(...)              Excel spec: Radar formulas, banding rules for junk / product, mappings

Conventions
-----------
- "N/A" (not applicable): the factor does not exist for that product / section. Numeric factors get a
  reserved value (default -1, or below the data minimum); text factors get "N/A". In Emblem, fix the
  N/A parameter at 0 (relativity 1.00) - see the report notes.
- "Unknown": genuinely missing where the factor does apply (Radar NoValue).
- Junk values (placeholders like 9999, "UNKNOWN", out-of-range) are set to missing here; in Radar they
  are mapped to the Unknown level by Banding rules (Radar formulas cannot create a NoValue).
"""

import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from banding import (C_EXPO, C_REF, HtmlReport, _cell, _esc, _fmt_any, _slug, _table, _weighted_quantiles,
                     candidate_edges, heat_table)

PRODUCT_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRING_PLACEHOLDERS = ["", "UNKNOWN", "UNK", "N/A", "NA", "NULL", "NONE", "?", "-", "TBC", "NOT KNOWN"]


# =============================================================================== engineering

def _is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def reserved_value(s, requested=None):
    """Numeric N/A code: requested, else -1 if all data >= 0, else one below the minimum."""
    if requested is not None:
        return requested
    v = s.dropna()
    return -1 if (len(v) == 0 or v.min() >= 0) else math.floor(v.min()) - 1


def applicable_mask(df, rule, product_col, sections):
    """True where the factor applies. rule = {"products": [...]} and/or {"section": "Buildings"}."""
    m = pd.Series(True, index=df.index)
    if "products" in rule:
        m &= df[product_col].astype("string").isin([str(p) for p in rule["products"]]).fillna(False)
    if "section" in rule:
        col = sections[rule["section"]]
        m &= df[col].astype("string").str.upper().str.strip().eq("Y").fillna(False)
    return m


def engineer(df, cfg):
    """
    Apply confirmed engineering to a copy of df. Returns (df, log) where log[factor] lists what changed.
    cfg keys used: EXPO, PRODUCT, SECTIONS, MAPPINGS, JUNK, STRING_PLACEHOLDERS, APPLIES_TO, CATEGORICAL
    """
    df = df.copy()
    expo = cfg["EXPO"]
    tot = df[expo].sum()
    log = {}
    found = {}                                   # placeholder text actually present, per factor

    def note(f, text):
        log.setdefault(f, []).append(text)

    # 1. mappings (raw -> harmonised), e.g. code lists that differ by product
    for f, mp in cfg.get("MAPPINGS", {}).items():
        if isinstance(mp, str):                                   # path to a CSV with columns raw,mapped
            t = pd.read_csv(mp, dtype=str)
            mp = dict(zip(t.iloc[:, 0], t.iloc[:, 1]))
        raw = df[f].astype("string")
        mapped = raw.map({str(k): v for k, v in mp.items()})
        unmapped = raw.notna() & mapped.isna()
        df[f] = mapped.where(~unmapped, raw)
        note(f, f"mapping applied ({len(mp)} entries); {int(unmapped.sum()):,} rows had no mapping and were kept as is")

    # 2. junk -> missing
    placeholders = [p.upper() for p in cfg.get("STRING_PLACEHOLDERS", STRING_PLACEHOLDERS)]
    for f in cfg["FACTORS"]:
        s = df[f]
        if not _is_num(s) and f not in cfg.get("NO_PLACEHOLDER_CHECK", []):
            hit = s.astype("string").str.strip().str.upper().isin(placeholders).fillna(False)
            if hit.any():
                found[f] = sorted(set(s[hit].astype(str).str.strip()))
                note(f, f"{int(hit.sum()):,} placeholder text values ({', '.join(sorted(set(s[hit].astype(str).str.strip())))[:80]}) "
                        f"set to missing ({df.loc[hit, expo].sum() / tot:.1%} of exposure)")
                df.loc[hit, f] = pd.NA
    for f, rule in cfg.get("JUNK", {}).items():
        s = pd.to_numeric(df[f], errors="coerce") if _is_num(df[f]) else df[f]
        hit = pd.Series(False, index=df.index)
        if "values" in rule:
            hit |= df[f].isin(rule["values"]) | df[f].astype("string").isin([str(v) for v in rule["values"]]).fillna(False)
        if "below" in rule:
            hit |= (s < rule["below"]).fillna(False)
        if "above" in rule:
            hit |= (s > rule["above"]).fillna(False)
        if hit.any():
            note(f, f"{int(hit.sum()):,} junk values set to missing ({df.loc[hit, expo].sum() / tot:.1%} of exposure): {rule}")
            df.loc[hit, f] = np.nan if _is_num(df[f]) else pd.NA

    # 3. not applicable -> N/A (reserved value for numbers, "N/A" for text)
    reserved = {}
    for f, rule in cfg.get("APPLIES_TO", {}).items():
        app = applicable_mask(df, rule, cfg["PRODUCT"], cfg.get("SECTIONS", {}))
        s = df[f]
        had_value = (~app & s.notna()).sum()
        if _is_num(s) and f not in cfg.get("CATEGORICAL", []):
            r = reserved_value(s[app], rule.get("reserved"))
            if (s[app] == r).any():
                raise ValueError(f"{f}: reserved value {r} occurs in real data — set 'reserved' in APPLIES_TO")
            df.loc[~app, f] = r
            reserved[f] = r
        else:
            df[f] = df[f].astype("string")
            df.loc[~app, f] = "N/A"
            reserved[f] = "N/A"
        unknown = (app & df[f].isna())
        note(f, f"N/A for {(~app).sum():,} rows ({df.loc[~app, expo].sum() / tot:.1%} of exposure), coded {reserved[f]!r}; "
                f"Unknown (missing where it applies): {df.loc[unknown, expo].sum() / max(df.loc[app, expo].sum(), 1e-9):.1%} "
                f"of applicable exposure"
                + (f"; note {had_value:,} N/A rows had a value that was overwritten" if had_value else ""))
    return df, log, reserved, found


# =============================================================================== detection

def _rate_by(df, flag, by, expo):
    g = df.groupby(by, dropna=False, observed=True).apply(
        lambda d: pd.Series({"exposure": d[expo].sum(), "rate": d.loc[flag.loc[d.index], expo].sum() / d[expo].sum()
                             if d[expo].sum() > 0 else np.nan}), include_groups=False)
    return g


def detect_structural(df, f, expo, product_col, products, sections, hi=0.95, lo=0.20, explained=0.80,
                      min_share=0.01):
    """
    Look for missing (or zero) values concentrated in some products or where a section is N.
    Returns a list of proposals: {"kind": "missing"/"zero", "rule": {...}, "evidence": text}.
    """
    s = df[f]
    tot = df[expo].sum()
    flags = {"missing": s.isna()}
    if _is_num(s):
        flags["zero"] = (s == 0).fillna(False)
    out = []
    for kind, flag in flags.items():
        if df.loc[flag, expo].sum() / tot < min_share:
            continue
        miss_tot = df.loc[flag, expo].sum()
        best = None
        # by product
        pr = df[product_col].astype("string")
        rows = []
        for p in products:
            m = pr.eq(p).fillna(False)
            e = df.loc[m, expo].sum()
            if e > 0:
                rows.append((p, df.loc[m & flag, expo].sum() / e, df.loc[m & flag, expo].sum() / miss_tot))
        absent = [p for p, r, _ in rows if r >= hi]
        present = [p for p, r, _ in rows if r <= lo]
        if absent and present and sum(x for p, r, x in rows if p in absent) >= explained \
                and len(absent) + len(present) == len(rows):
            share = sum(x for p, r, x in rows if p in absent)
            best = {"kind": kind, "rule": {"products": present}, "score": share,
                    "evidence": "; ".join(f"{p}: {r:.0%} {kind}" for p, r, _ in rows)}
        # by section
        for name, col in sections.items():
            n = df[col].astype("string").str.upper().str.strip().eq("N").fillna(False)
            y = df[col].astype("string").str.upper().str.strip().eq("Y").fillna(False)
            en, ey = df.loc[n, expo].sum(), df.loc[y, expo].sum()
            if en <= 0 or ey <= 0:
                continue
            rn, ry = df.loc[n & flag, expo].sum() / en, df.loc[y & flag, expo].sum() / ey
            share = df.loc[n & flag, expo].sum() / miss_tot
            if rn >= hi and ry <= lo and share >= explained and (best is None or share > best["score"]):
                best = {"kind": kind, "rule": {"section": name}, "score": share,
                        "evidence": f"{name} cover N: {rn:.0%} {kind}; {name} cover Y: {ry:.0%} {kind}"}
        if best:
            best["explains"] = best.pop("score")
            out.append(best)
    return out


def _nines(v):
    return v >= 99 and re.fullmatch(r"9+", str(int(v))) is not None if float(v).is_integer() else False


def factor_checks(df_raw, df_eng, f, cfg, reserved):
    """All checks for one factor. Returns dict of results (flags are (level, text) pairs)."""
    expo, prod, products = cfg["EXPO"], cfg["PRODUCT"], cfg["PRODUCTS"]
    tot = df_raw[expo].sum()
    s_raw, s = df_raw[f], df_eng[f]
    cat = f in cfg.get("CATEGORICAL", []) or not _is_num(s)
    r = reserved.get(f)
    applicable = ~(s == r).fillna(False) if r is not None else pd.Series(True, index=s.index)
    sa = s[applicable]
    flags = []
    res = {"factor": f, "type": "categorical" if cat else "numeric", "distinct": int(s_raw.nunique()),
           "missing_raw": df_raw.loc[s_raw.isna(), expo].sum() / tot,
           "na_share": df_eng.loc[~applicable, expo].sum() / tot if r is not None else 0.0,
           "unknown": df_eng.loc[applicable & s.isna(), expo].sum() / max(df_eng.loc[applicable, expo].sum(), 1e-9)}

    # structural missing / zero (only suggest if not already handled)
    props = [] if f in cfg.get("APPLIES_TO", {}) else detect_structural(
        df_raw, f, expo, prod, products, cfg.get("SECTIONS", {}))
    res["proposals"] = props
    for p in props:
        flags.append(("action", f"{p['kind'].capitalize()} values look structural ({p['explains']:.0%} of them explained): "
                                f"{p['evidence']}. Proposed: treat as not applicable outside {_rule_text(p['rule'])}."))
    if r is None and res["missing_raw"] >= 0.03 and not any(p["kind"] == "missing" for p in props):
        flags.append(("check", f"{res['missing_raw']:.0%} of exposure missing, not explained by product or section"))

    # missing rate drifting over time
    per = cfg.get("PERIOD")
    if per and s.isna().any():
        by = df_eng.loc[applicable].groupby(per)[[expo]].sum()
        by["m"] = df_eng.loc[applicable & s.isna()].groupby(per)[expo].sum()
        rate = (by["m"].fillna(0) / by[expo])
        if len(rate) > 1 and rate.max() - rate.min() >= 0.2:
            flags.append(("check", f"missing rate changes over time ({rate.min():.0%} to {rate.max():.0%} by {per})"))

    # junk candidates (numeric): all-9 placeholders, negatives in a non-negative field
    if not cat:
        v = pd.to_numeric(sa, errors="coerce").dropna()
        w = df_eng.loc[v.index, expo]
        nines = sorted({x for x in v.unique() if _nines(x) and x >= v.quantile(0.99)})
        if nines and f not in cfg.get("JUNK", {}):
            sh = w[v.isin(nines)].sum() / tot
            flags.append(("action", f"possible placeholder values {', '.join(_fmt_any(x) for x in nines)} "
                                    f"({sh:.1%} of exposure). If junk, add to JUNK: \"{f}\": {{\"values\": {[_tidy(x) for x in nines]}}}"))
        neg = w[v < 0].sum() / tot
        if 0 < neg and (v >= 0).mean() >= 0.99 and f not in cfg.get("JUNK", {}):
            flags.append(("action", f"negative values in a mostly non-negative field ({neg:.2%} of exposure)"))
        # spikes (possible defaults)
        if v.nunique() > 20:
            top = w.groupby(v).sum().sort_values(ascending=False)
            share = top / w.sum()
            spikes = share[share >= 0.03]
            zero_done = any(p["kind"] == "zero" for p in props)
            for val, sh in spikes.items():
                if val == 0 and zero_done:
                    continue
                where = "the minimum" if val == v.min() else "a single value"
                flags.append(("check", f"{sh:.0%} of exposure at {where} {_fmt_any(val)}: possible default or rounded input"))
        res["percentiles"] = _pct_by_product(df_eng.loc[v.index], f, expo, prod, products)
        # overlap: a product's median outside the rest's 5th-95th range
        for p in products:
            m = df_eng.loc[v.index, prod].astype("string").eq(p).fillna(False)
            if m.sum() > 50 and (~m).sum() > 50:
                med = _weighted_quantiles(v[m].to_numpy(float), w[m].to_numpy(float), [0.5])[0]
                lo_, hi_ = _weighted_quantiles(v[~m].to_numpy(float), w[~m].to_numpy(float), [0.05, 0.95])
                if med < lo_ or med > hi_:
                    flags.append(("check", f"little overlap: {p}'s median {_fmt_any(med)} is outside the other "
                                           f"products' 5th–95th range ({_fmt_any(lo_)}–{_fmt_any(hi_)})"))
    else:
        lv = sa.dropna().astype(str)
        res["levels"] = int(lv.nunique())
        if res["levels"] > 252:
            flags.append(("action", f"{res['levels']} levels: Emblem allows 255 per factor (including Missing/N/A/OTHER) "
                                    f"— needs grouping (parent group or OTHER) before Emblem"))
        bad = s_raw.astype("string").str.strip().str.upper().isin(
            [x.upper() for x in cfg.get("STRING_PLACEHOLDERS", STRING_PLACEHOLDERS)]).fillna(False)
        if bad.any():
            flags.append(("info", f"placeholder text ({df_raw.loc[bad, expo].sum() / tot:.1%} of exposure) set to missing"))
        # levels that only exist in one product
        d = df_eng.loc[lv.index]
        ct = d.groupby([lv, d[prod].astype("string")])[expo].sum().unstack(fill_value=0)
        lvl_tot = ct.sum(axis=1)
        single = ct.div(lvl_tot, axis=0).max(axis=1) >= 0.98
        sh = lvl_tot[single].sum() / lvl_tot.sum() if lvl_tot.sum() else 0
        if 0.1 <= sh < 0.999 and ct.shape[1] > 1:
            flags.append(("check", f"{single.sum()} levels ({sh:.0%} of exposure) appear in only one product — "
                                   f"their effect can't be separated from that product's"))
        res["level_mix"] = ct
    res["flags"] = flags
    return res


def _tidy(x):
    x = float(x)
    return int(x) if x.is_integer() else x


def _rule_text(rule):
    parts = []
    if "products" in rule:
        parts.append("products " + ", ".join(rule["products"]))
    if "section" in rule:
        parts.append(f"{rule['section']} cover = Y")
    return " and ".join(parts)


def _pct_by_product(d, f, expo, prod, products):
    rows = {}
    for p in ["All"] + list(products):
        m = slice(None) if p == "All" else d[prod].astype("string").eq(p).fillna(False)
        v, w = pd.to_numeric(d.loc[m, f], errors="coerce"), d.loc[m, expo]
        ok = v.notna()
        if ok.sum() < 5:
            continue
        q = _weighted_quantiles(v[ok].to_numpy(float), w[ok].to_numpy(float), [0.05, 0.25, 0.5, 0.75, 0.95])
        rows[p] = {"exposure": w.sum(), "p05": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4]}
    return pd.DataFrame(rows).T


# =============================================================================== product / cover

def product_summary(df, cfg, perils):
    expo, prod, per = cfg["EXPO"], cfg["PRODUCT"], cfg.get("PERIOD")
    p = df[prod].astype("string").fillna("(blank)")
    known = set(cfg["PRODUCTS"])
    unexpected = sorted(set(p.unique()) - known)
    cost = sum(df[k] for _, k in perils.values())
    t = pd.DataFrame({"rows": p.value_counts()})
    t["exposure"] = df.groupby(p)[expo].sum()
    t["expo_share"] = t.exposure / t.exposure.sum()
    for name, (cc, _) in perils.items():
        t[f"claims_{name}"] = df.groupby(p)[cc].sum()
    t["burning_cost"] = cost.groupby(p).sum() / t.exposure
    t = t.loc[[x for x in cfg["PRODUCTS"] if x in t.index] + [x for x in t.index if x not in known]]
    by_year = df.groupby([per, p])[expo].sum().unstack(fill_value=0) if per else None
    return t, unexpected, by_year


def section_mix(df, cfg):
    expo, prod, sections = cfg["EXPO"], cfg["PRODUCT"], cfg.get("SECTIONS", {})
    if not sections:
        return None
    key = df[list(sections.values())].astype("string").fillna("?").apply(
        lambda r: " / ".join(f"{n} {v}" for n, v in zip(sections, r)), axis=1)
    t = df.groupby([key, df[prod].astype("string")])[expo].sum().unstack(fill_value=0)
    return t.div(t.sum(axis=0), axis=1)        # share of each product's exposure


def cover_check(df, cfg, perils, min_expected=10, ratio=0.1, min_share=0.005):
    """Claims vs expected (at the overall frequency) by product x section combination x peril."""
    expo, prod, sections = cfg["EXPO"], cfg["PRODUCT"], cfg.get("SECTIONS", {})
    keys = [df[prod].astype("string").rename("product")] + [
        df[c].astype("string").str.upper().str.strip().rename(n) for n, c in sections.items()]
    g = df.groupby(keys, dropna=False)
    e = g[expo].sum()
    keep = e / e.sum() >= min_share
    ae, flags = {}, []
    for name, (cc, _) in perils.items():
        freq = df[cc].sum() / df[expo].sum()
        act = g[cc].sum()
        exp_ = e * freq
        ae[name] = (act / exp_)[keep]
        for idx in exp_[keep].index:
            if exp_[idx] >= min_expected and act[idx] <= ratio * exp_[idx]:
                lab = idx if isinstance(idx, str) else " / ".join(f"{k}={v}" for k, v in zip(["product"] + list(sections), idx))
                flags.append(f"{name}: {lab} has {int(act[idx])} claims vs {exp_[idx]:.0f} expected — possibly not covered")
    t = pd.DataFrame(ae)
    t.index = [" / ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in t.index]
    t.insert(0, "exposure share", (e / e.sum())[keep].to_numpy())
    return t, flags


# =============================================================================== charts

def _colours(products):
    return {p: PRODUCT_COLOURS[i % len(PRODUCT_COLOURS)] for i, p in enumerate(products)}


def _layout(fig, title, ytitle, height=400, yfmt=None):
    fig.update_layout(title=dict(text=title, y=0.97), template="plotly_white", hovermode="x unified", height=height,
                      margin=dict(t=90, b=20, l=70, r=40), yaxis_title=ytitle,
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    fig.update_xaxes(type="category", automargin=True)
    return fig


def plot_missing_by_year(df, f, cfg, colours):
    expo, prod, per = cfg["EXPO"], cfg["PRODUCT"], cfg["PERIOD"]
    miss = df[f].isna()
    fig = go.Figure()
    for p in cfg["PRODUCTS"]:
        m = df[prod].astype("string").eq(p).fillna(False)
        e = df[m].groupby(per)[expo].sum()
        mm = df[m & miss].groupby(per)[expo].sum().reindex(e.index, fill_value=0)
        fig.add_trace(go.Scatter(x=[str(_tidy(i)) for i in e.index], y=mm / e, name=p, mode="lines+markers",
                                 line=dict(color=colours[p], width=2), marker=dict(size=7),
                                 hovertemplate=f"{p}: %{{y:.0%}}<extra></extra>"))
    return _layout(fig, f"{f} — share of exposure missing (raw data), by {per} and product", "Missing", 360, ".0%")


def _fine_codes(df, f, cfg, reserved, categorical):
    """Fine grid labels per row for the readiness one-ways: N/A and Missing as their own levels."""
    s, expo = df[f], cfg["EXPO"]
    r = reserved.get(f)
    lab = pd.Series(pd.NA, index=s.index, dtype="object")
    na = (s == r).fillna(False) if r is not None else pd.Series(False, index=s.index)
    if categorical:
        top = df.loc[~na].groupby(s[~na].astype(str))[expo].sum().sort_values(ascending=False).index[:25]
        st = s.astype("string")
        lab = st.where(st.isin(top), "(other levels)")
        order = list(top) + ["(other levels)"]
    else:
        sub = df.loc[~na & s.notna()]
        edges, whole = candidate_edges(sub, f, 15, expo)
        codes = pd.cut(s.where(~na), [-np.inf, *edges, np.inf], right=False, labels=False)
        from banding import numeric_labels
        names = numeric_labels(edges, False, whole, _tidy(sub[f].min()) if whole and len(sub) else None)
        lab = codes.map(lambda c: names[int(c)] if pd.notna(c) else pd.NA)
        order = names
    lab = lab.astype("object")
    lab[s.isna()] = "Missing"
    lab[na] = "N/A"
    order = order + ["Missing", "N/A"]
    return lab, [o for o in order if (lab == o).any()]


def plot_mix(df, f, cfg, reserved, categorical, colours):
    """Share of each product's exposure across the fine levels — shows where products overlap."""
    expo, prod = cfg["EXPO"], cfg["PRODUCT"]
    lab, order = _fine_codes(df, f, cfg, reserved, categorical)
    t = df.groupby([lab, df[prod].astype("string")])[expo].sum().unstack(fill_value=0).reindex(order)
    share = t / t.sum(axis=0)
    fig = go.Figure()
    for p in cfg["PRODUCTS"]:
        if p in share:
            fig.add_trace(go.Scatter(x=order, y=share[p], name=p, mode="lines+markers",
                                     line=dict(color=colours[p], width=2, shape="linear"), marker=dict(size=7),
                                     hovertemplate=f"{p}: %{{y:.1%}}<extra></extra>"))
    fig = _layout(fig, f"{f} — where each product's exposure sits", "Share of product's exposure", 380, ".0%")
    fig.update_xaxes(tickangle=-40 if len(order) > 8 else 0)
    return fig


def plot_oneway_by_product(df, f, cfg, reserved, categorical, colours, cost, claims, min_claims=30):
    """Burning cost relativity (all perils, 1 = overall average) by product on the fine levels."""
    expo, prod = cfg["EXPO"], cfg["PRODUCT"]
    lab, order = _fine_codes(df, f, cfg, reserved, categorical)
    d = pd.DataFrame({"lab": lab, "p": df[prod].astype("string"), "e": df[expo], "c": cost, "n": claims})
    avg = d.c.sum() / d.e.sum()
    g = d.groupby(["lab", "p"])[["e", "c", "n"]].sum()
    fig = go.Figure()
    for p in cfg["PRODUCTS"]:
        if p not in g.index.get_level_values(1):
            continue
        t = g.xs(p, level=1).reindex(order)
        rel = (t.c / t.e) / avg
        thin = (t.n.fillna(0) < min_claims).to_numpy()
        fig.add_trace(go.Scatter(x=order, y=rel, name=p, mode="lines+markers", connectgaps=False,
                                 line=dict(color=colours[p], width=2),
                                 marker=dict(size=8, color=colours[p], symbol=np.where(thin, "circle-open", "circle")),
                                 customdata=t.n.fillna(0).to_numpy(),
                                 hovertemplate=f"{p}: %{{y:.2f}} (%{{customdata:,.0f}} claims)<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=C_REF, width=1, dash="dot"))
    fig = _layout(fig, f"{f} — burning cost relativity by product (all perils, 1 = overall average)",
                  "Relativity", 420)
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(tickangle=-40 if len(order) > 8 else 0)
    return fig


# =============================================================================== report

def radar_formula(f, rule, reserved, cfg, radar_names=None):
    """Radar Formula component text for an N/A recode."""
    rn = radar_names or {}
    v = rn.get(f, f)
    conds = []
    if "products" in rule:
        prod = rn.get(cfg["PRODUCT"], cfg["PRODUCT"])
        # not applicable = any product outside the list (including UNKNOWN / blank)
        c = " And ".join(f'{prod} <> "{p}"' for p in rule["products"])
        conds.append(f"({c})" if len(rule["products"]) > 1 and "section" in rule else c)
    if "section" in rule:
        sc = rn.get(cfg["SECTIONS"][rule["section"]], cfg["SECTIONS"][rule["section"]])
        conds.append(f'{sc} <> "Y"')
    cond = " Or ".join(c for c in conds if c)
    val = f'"{reserved}"' if isinstance(reserved, str) else _fmt_plain(reserved)
    return f"{v}Model = If {cond} then {val} else {v};"


def _fmt_plain(x):
    x = float(x)
    return str(int(x)) if x.is_integer() else repr(x)


def build_report(df_raw, df_eng, cfg, perils, checks, eng_log, reserved, prod_tab, unexpected, by_year, mix,
                 cover_tab, cover_flags, notes, path):
    expo = cfg["EXPO"]
    colours = _colours(cfg["PRODUCTS"])
    cost = sum(df_eng[k] for _, k in perils.values())
    claims = sum(df_eng[c] for c, _ in perils.values())
    rep = HtmlReport(title=f"Data readiness {cfg['VERSION']}",
                     subtitle=f"{cfg['SOURCE_NAME']} — checks and feature engineering before screening and banding")
    rep.meta = {"Source": cfg["SOURCE"], "Rows": f"{len(df_raw):,}", "Exposure": f"{df_raw[expo].sum():,.0f}",
                "Claims (all perils)": f"{claims.sum():,.0f}", "Factors checked": f"{len(checks)}"}
    rep.checks = notes

    # ---- headline: products
    unexpected_html = ("<p class='note'><b>Products outside the expected list:</b> "
                       + ", ".join(_esc(u) for u in unexpected) + " — these fall to UNKNOWN in the Radar rules.</p>"
                       if unexpected else "<p class='note'>No products outside "
                       + ", ".join(cfg["PRODUCTS"]) + " — the UNKNOWN catch-all is not triggered.</p>")
    fmt = {"rows": ",.0f", "exposure": ",.0f", "expo_share": ".1%", "burning_cost": ",.2f"}
    fmt.update({c: ",.0f" for c in prod_tab.columns if c.startswith("claims_")})
    hdr = {"rows": "Rows", "exposure": "Exposure", "expo_share": "Exposure %", "burning_cost": "Burning cost"}
    hdr.update({c: c.replace("claims_", "Claims ") for c in prod_tab.columns if c.startswith("claims_")})
    html = "<h2>Products</h2>" + _table(prod_tab, fmt, hdr, index_name="Product") + unexpected_html
    if by_year is not None:
        by = by_year.copy()
        by.index = [str(_tidy(i)) for i in by.index]
        html += "<h3>Exposure by year</h3>" + _table(by, {c: ",.0f" for c in by.columns}, index_name=cfg["PERIOD"])
    if mix is not None:
        html += ("<h3>Section mix (share of each product's exposure)</h3>"
                 + _table(mix, {c: ".1%" for c in mix.columns}, index_name="Cover"))
    # ---- cover check
    html += ("<h2>Peril cover check</h2><p class='note'>Actual ÷ expected claims for each peril, where expected uses "
             "the overall frequency for that peril. Very low values with plenty of expected claims suggest the "
             "peril isn't covered for that combination. Nothing is excluded automatically.</p>"
             + heat_table(cover_tab.drop(columns="exposure share"), "Product / cover", fmt=".2f"))
    html += ("<ul class='checks'>" + "".join(f"<li>{_esc(x)}</li>" for x in cover_flags) + "</ul>"
             if cover_flags else "<p class='note'>No product / cover combinations look uncovered.</p>")

    # ---- factor overview
    rows = []
    for c in checks:
        acts = sum(1 for lv, _ in c["flags"] if lv == "action")
        chk = sum(1 for lv, _ in c["flags"] if lv == "check")
        status = ("engineered" if c["factor"] in cfg.get("APPLIES_TO", {}) else "")
        rows.append({"factor": c["factor"], "type": c["type"], "distinct": c["distinct"],
                     "missing (raw)": c["missing_raw"], "N/A": c["na_share"], "Unknown": c["unknown"],
                     "actions": acts, "checks": chk, "status": status})
    ov = pd.DataFrame(rows).set_index("factor").sort_values(["actions", "checks"], ascending=False)
    html += ("<h2>Factors</h2><p class='note'><b>Actions</b> need a decision before screening or banding "
             "(paste the proposed line into the config). <b>Checks</b> are worth a look. "
             "N/A = not applicable (engineered); Unknown = missing where the factor applies, as a share of "
             "applicable exposure.</p>"
             + _table(ov, {"missing (raw)": ".1%", "N/A": ".1%", "Unknown": ".1%"},
                      {"type": "Type", "distinct": "Distinct", "missing (raw)": "Missing (raw)", "actions": "Actions",
                       "checks": "Checks", "status": "Status"}, index_name="Factor",
                      links={i: _slug(i) for i in ov.index}))
    rep.summary_html = html

    # ---- per factor
    lvl_word = {"action": "Action", "check": "Check", "info": "Info"}
    for c in checks:
        f = c["factor"]
        cat = c["type"] == "categorical"
        flag_html = "".join(f"<li><b>{lvl_word[lv]}:</b> {_esc(t)}</li>" for lv, t in c["flags"]) or "<li>No issues found.</li>"
        eng_html = "".join(f"<li>{_esc(t)}</li>" for t in eng_log.get(f, []))
        intro = (f"<div class='stats'>"
                 + "".join(f"<div class='stat'><span>{k}</span><b>{_esc(v)}</b></div>" for k, v in [
                     ("Type", c["type"]), ("Distinct", f"{c['distinct']:,}"), ("Missing (raw)", f"{c['missing_raw']:.1%}"),
                     ("N/A", f"{c['na_share']:.1%}"), ("Unknown", f"{c['unknown']:.1%}")])
                 + "</div><ul class='checks'>" + flag_html + "</ul>"
                 + (f"<h3>Engineering applied</h3><ul class='checks'>{eng_html}</ul>" if eng_html else ""))
        for p in c["proposals"]:
            line = f'"{f}": {_json_rule(p["rule"])},'
            intro += ("<p class='note'>To apply the proposed recode, add this line to APPLIES_TO and run again:</p>"
                      f"<pre>{_esc(line)}</pre>")
        if f in cfg.get("APPLIES_TO", {}):
            intro += ("<p class='note'>Radar formula (add before the Banding component):</p><pre>"
                      + _esc(radar_formula(f, cfg["APPLIES_TO"][f], reserved[f], cfg, cfg.get("RADAR_NAMES")))
                      + "</pre><p class='note'>In Emblem, fix the N/A level's parameter at 0 (relativity 1.00) using "
                        "Fixing Factors, so the product / section relativity carries those policies and this factor "
                        "only adjusts the policies it applies to. Choose the base level from the applicable levels.</p>")
        blocks = []
        if c.get("percentiles") is not None and len(c["percentiles"]):
            pt = c["percentiles"]
            blocks.append(("Distribution by product (exposure-weighted, applicable rows)",
                           _table(pt, {k: (",.0f" if k == "exposure" else None) for k in pt.columns}, index_name="Product"),
                           ""))
        if cat and c.get("level_mix") is not None:
            lm = c["level_mix"]
            top = lm.sum(axis=1).sort_values(ascending=False).index[:30]
            share = lm.loc[top].div(lm.loc[top].sum(axis=1), axis=0)
            blocks.append(("Level mix by product (largest 30 levels)",
                           heat_table(share, "Level", fmt=".0%"),
                           "Share of each level's exposure by product. Rows close to 100% in one column are levels "
                           "that only exist for that product."))
        if cfg.get("PERIOD") and df_raw[f].isna().any():
            blocks.append(("Missing by year", plot_missing_by_year(df_raw, f, cfg, colours), ""))
        blocks.append(("Where each product's exposure sits", plot_mix(df_eng, f, cfg, reserved, cat, colours),
                       "Engineered values. Missing and N/A shown as their own levels."))
        blocks.append(("Burning cost by product", plot_oneway_by_product(df_eng, f, cfg, reserved, cat, colours, cost, claims),
                       "One-way only. Lines that run parallel mean the factor behaves the same way in each product; "
                       "lines that cross or diverge suggest a product interaction to test in Emblem. "
                       "Hollow markers have fewer than 30 claims."))
        rep.add_block_section(f, blocks, intro)
    return rep.save(path)


def _json_rule(rule):
    import json
    return json.dumps({k: v for k, v in rule.items()})


# =============================================================================== spec workbook

def write_spec(path, cfg, checks, reserved, eng_log, found=None, df_eng=None):
    """Excel spec for Radar: formulas, junk/product banding rules, mappings, factor checks."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    head = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="2A78D6")

    def sheet(title, header, rows, widths=None, first=False):
        ws = wb.active if first else wb.create_sheet(title)
        ws.title = title
        ws.append(header)
        for c in ws[1]:
            c.font, c.fill = head, fill
        for r in rows:
            ws.append(list(r))
        for i, w in enumerate(widths or [], start=1):
            ws.column_dimensions[get_column_letter(i)].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        return ws

    sheet("Read me", ["Step", "What to do in Radar"], [
        ("1", "Product banding: add the rules on the 'Product rules' sheet. RPO/CPO/RE map to themselves; NoValue and "
              "Default map to UNKNOWN as a catch-all. The Radar guide (section 34.3) warns that Default rules can hide new "
              "levels after a data refresh, so check the UNKNOWN count is zero after each refresh."),
        ("2", "Mappings: for factors on the 'Mappings' sheet, read the mapping CSV as a data source and join on the raw "
              "value (needed where there are too many levels to type rules)."),
        ("3", "Not-applicable recodes: add the formulas on the 'Radar formulas' sheet in a Formula component before the "
              "Banding component. Each creates <Factor>Model with a reserved value for N/A."),
        ("4", "Junk values: in the Banding component, add the rules on the 'Junk rules' sheet so placeholder / "
              "out-of-range values go to the Unknown level along with NoValue. Radar formulas can't create NoValue, "
              "so this is done in Banding."),
        ("5", "N/A level: add a rule '= <reserved value>' -> N/A (numeric) or '= \"N/A\"' -> N/A (text) for each recoded "
              "factor. In Emblem, fix the N/A parameter at 0 (Fixing Factors) and pick the base level from the other levels."),
        ("6", "Tidy banding of the remaining values follows in a later step (after screening)."),
    ], [8, 120], first=True)

    sheet("Factor checks", ["Factor", "Type", "Distinct", "Missing (raw)", "N/A", "Unknown", "Level", "Finding"],
          [(c["factor"], c["type"], c["distinct"], f"{c['missing_raw']:.1%}", f"{c['na_share']:.1%}",
            f"{c['unknown']:.1%}", lv, t) for c in checks for lv, t in (c["flags"] or [("", "No issues found")])],
          [22, 12, 10, 12, 8, 10, 8, 110])

    frows = []
    for f, rule in cfg.get("APPLIES_TO", {}).items():
        frows.append((f, _rule_text(rule), reserved[f], radar_formula(f, rule, reserved[f], cfg, cfg.get("RADAR_NAMES")),
                      "; ".join(eng_log.get(f, []))))
    sheet("Radar formulas", ["Factor", "Applies to", "N/A code", "Formula (Formula component)", "Effect in the data"],
          frows or [("(none confirmed yet)", "", "", "", "")], [22, 30, 10, 80, 90])

    jrows = []
    for f, rule in cfg.get("JUNK", {}).items():
        for v in rule.get("values", []):
            jrows.append((f, "=", v, "", "", "Unknown", "placeholder value"))
        if "below" in rule:
            jrows.append((f, "<", rule["below"], "", "", "Unknown", "below valid range"))
        if "above" in rule:
            jrows.append((f, ">", rule["above"], "", "", "Unknown", "above valid range"))
    for f, vals in (found or {}).items():
        for v in vals:
            if v != "":
                jrows.append((f, "=", v, "", "", "Unknown", "placeholder text"))
    for c in checks:
        f = c["factor"]
        if df_eng is not None and df_eng[f].isna().any():
            jrows.append((f, "NoValue", "", "", "", "Unknown", "blank / missing where the factor applies"))
    sheet("Junk rules", ["Factor", "Lower Operator", "Lower Bound", "Upper Operator", "Upper Bound", "Level name", "Why"],
          jrows or [("(none)", "", "", "", "", "", "")], [22, 14, 14, 14, 14, 12, 60])

    prows = [("=", p, "", "", p) for p in cfg["PRODUCTS"]] + [("NoValue", "", "", "", "UNKNOWN"),
                                                               ("Default", "", "", "", "UNKNOWN")]
    sheet("Product rules", ["Lower Operator", "Lower Bound", "Upper Operator", "Upper Bound", "Level name"], prows,
          [14, 14, 14, 14, 14])

    mrows = []
    for f, mp in cfg.get("MAPPINGS", {}).items():
        if isinstance(mp, str):
            t = pd.read_csv(mp, dtype=str)
            mp = dict(zip(t.iloc[:, 0], t.iloc[:, 1]))
        mrows += [(f, k, v) for k, v in mp.items()]
    sheet("Mappings", ["Factor", "Raw value", "Mapped value"], mrows or [("(none)", "", "")], [22, 30, 30])
    wb.save(path)
    return path
