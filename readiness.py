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
from banding import _fmt_any, _weighted_quantiles, find_points, year_like


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

    # 2. junk -> missing; infinite values (e.g. a ratio divided by zero) are missing too
    for f in cfg["FACTORS"]:
        if f not in cfg.get("CATEGORICAL", []) and pd.api.types.is_float_dtype(df[f]):
            inf = np.isinf(df[f].to_numpy(dtype=float, na_value=np.nan))
            if inf.any():
                note(f, f"{int(inf.sum()):,} infinite values (e.g. divided by zero) set to missing "
                        f"({df.loc[inf, expo].sum() / tot:.1%} of exposure)")
                df.loc[inf, f] = np.nan
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
                + (f"; {had_value:,} of the N/A rows held a value (e.g. 0), now replaced by the N/A code" if had_value else ""))
    return df, log, reserved, found


# =============================================================================== detection


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
        commas = not (f in cfg.get("NO_COMMAS", []) or year_like(v, w))     # years: 2026, not 2,026

        def fa(x):
            return _fmt_any(x, commas)
        nines = sorted({x for x in v.unique() if _nines(x) and x >= v.quantile(0.99)})
        if nines and f not in cfg.get("JUNK", {}):
            sh = w[v.isin(nines)].sum() / tot
            flags.append(("action", f"possible placeholder values {', '.join(fa(x) for x in nines)} "
                                    f"({sh:.1%} of exposure). If junk, add to JUNK: \"{f}\": {{\"values\": {[_tidy(x) for x in nines]}}}"))
        neg = w[v < 0].sum() / tot
        if 0 < neg and (v >= 0).mean() >= 0.99 and f not in cfg.get("JUNK", {}):
            flags.append(("action", f"negative values in a mostly non-negative field ({neg:.2%} of exposure)"))
        # spikes: the same rule that keeps a value as its own level (find_points), so the check and the
        # levels always agree; only factors with over 20 distinct values, where a lone spike stands out
        if v.nunique() > 20:
            share = w.groupby(v).sum() / w.sum()
            zero_done = any(p["kind"] == "zero" for p in props)
            for val, _ in find_points(v.astype(float), w.astype(float), cfg.get("POINT_MIN", 0.02),
                                      cfg.get("POINT_RATIO", 3)):
                if val == 0 and zero_done:
                    continue
                where = "the minimum" if val == v.min() else "a single value"
                flags.append(("check", f"{share[val]:.0%} of exposure at {where} {fa(val)}: a spike, kept as its own "
                                       f"level - check it isn't a default or rounded input"))
        res["percentiles"] = _pct_by_product(df_eng.loc[v.index], f, expo, prod, products)
        # overlap: a product's median outside the rest's 5th-95th range
        for p in products:
            m = df_eng.loc[v.index, prod].astype("string").eq(p).fillna(False)
            if m.sum() > 50 and (~m).sum() > 50:
                med = _weighted_quantiles(v[m].to_numpy(float), w[m].to_numpy(float), [0.5])[0]
                lo_, hi_ = _weighted_quantiles(v[~m].to_numpy(float), w[~m].to_numpy(float), [0.05, 0.95])
                if med < lo_ or med > hi_:
                    flags.append(("check", f"little overlap: {p}'s median {fa(med)} is outside the other "
                                           f"products' 5th–95th range ({fa(lo_)}–{fa(hi_)})"))
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


# =============================================================================== report


def _json_rule(rule):
    import json
    return json.dumps({k: v for k, v in rule.items()})


# =============================================================================== spec workbook

