"""
Pre-modelling review — data checks, raw data, suggested Radar levels, indicative Emblem groups,
feature importance and correlation between factors.

How to run (banding.py, readiness.py, review.py and importance.py must be in the same folder):
    Command prompt:   python run_review.py
    Jupyter:          %run run_review.py

One-off setup:
    pip install pandas plotly pyreadstat openpyxl scikit-learn lightgbm
    (lightgbm is optional: without it, feature importance uses scikit-learn)

Outputs (in OUT_DIR, all tagged with VERSION; switch each on/off in OUTPUTS):
    review_ALL_v1.html       data checks, importance, correlation + every factor on all perils combined
    review_<PERIL>_v1.html   every factor for one peril (same layout)
    radar_spec_v1.xlsx       Radar set-up: product rules, formulas, banding rules, level order, base levels
    tidy_map_<factor>_v1.csv long categorical mappings to join in Radar instead of typing rules

Each factor shows three charts (exposure bars stacked by product, relativity lines):
    1. Raw data   2. Suggested levels for modelling (Radar)   3. Indicative Emblem grouping
Turn any part of the reports off in REPORT_SECTIONS.

Workflow:
    1. Edit CONFIG, run, open review_ALL. Resolve the "actions" by pasting the proposed lines into
       APPLIES_TO / JUNK and running again.
    2. Check the suggested levels in each peril's report; change any with TIDY_OVERRIDES.
    3. Set up Radar from radar_spec. Final grouping happens in Emblem.
"""

import time
from pathlib import Path

import pandas as pd

from banding import HtmlReport, _esc, load_data, prepare
from readiness import (_colours, cover_check, engineer, factor_checks, product_summary, reserved_value,
                       section_mix)
from review import (add_factor_section, cover_html, factor_flags, factor_table_html, factor_views, level_series,
                    products_html, radar_formula, raw_spec, tidy_spec, variate_x, write_spec, _real)

# =============================================================================== CONFIG

SAS_FILE = r"Z:\Pricing\SME\Modelling\sme_property_2026.sas7bdat"
OUT_DIR = r"Z:\Pricing\SME\Modelling\review"
VERSION = "v1"

EXPO = "exposure"
PERIOD = "policy_year"          # None if you don't have one

PRODUCT = "product"
PRODUCTS = ["RPO", "CPO", "RE"]           # anything else (or blank) is reported and falls to UNKNOWN in Radar

SECTIONS = {"Buildings": "Buildings_Cover", "Contents": "Contents_Cover"}     # Y/N cover flags

PERILS = ["AD", "Fire", "EoW", "Theft", "Storm", "Flood"]
COUNT_COL = "No_Of_Claims_{peril}"
COST_COL = "Claim_Cost_{peril}"

FACTORS = ["sum_insured_buildings", "sum_insured_contents", "turnover", "years_trading", "employees",
           "trade_code", "construction", "occupancy"]
CATEGORICAL = ["trade_code", "construction", "occupancy"]
KEEP_ORDER_AS_LISTED = True     # False: factors appear in order of relativity spread

# ---- what to produce: True = on, False = off ----
OUTPUTS = dict(
    all_report=True,            # review_ALL: all perils combined
    peril_reports=True,         # review_<PERIL>: one per peril
    radar_spec=True,            # radar_spec workbook (+ tidy_map CSVs)
)
REPORT_SECTIONS = dict(
    data_checks=True,           # data-preparation notes; products and exposure by year (ALL report)
    cover_check=True,           # section mix and peril cover check (ALL report)
    factor_table=True,          # the table of factors at the top
    importance=True,            # feature importance (gradient boosting) - the slowest part
    correlation=True,           # correlation between factors (ALL report)
    factor_details=True,        # each factor: summary tiles, data findings, engineering, Radar formula
    raw_view=True,              # chart 1: raw data
    tidy_view=True,             # chart 2: suggested levels for modelling (Radar)
    group_view=True,            # chart 3: indicative Emblem grouping
    stability=True,             # stability by year (collapsed by default)
    tables=True,                # tables under charts 2 and 3
)

# ---- data fixes you've confirmed (paste proposed lines from the report) ----
APPLIES_TO = {                  # not applicable outside these products / where this section is N
    # "years_trading": {"products": ["CPO", "RE"]},
    # "sum_insured_buildings": {"section": "Buildings"},
}
JUNK = {                        # placeholder / out-of-range values -> Unknown
    # "employees": {"values": [9999]},
    # "years_trading": {"above": 150},
}
STRING_PLACEHOLDERS = ["", "UNKNOWN", "UNK", "N/A", "NA", "NULL", "NONE", "?", "-", "TBC", "NOT KNOWN"]
MAPPINGS = {}                   # {"factor": {"raw": "mapped"}} or {"factor": "mapping.csv"}

# ---- views and suggested modelling levels (Radar) - the same for every peril ----
TIDY = dict(
    max_levels=250,             # Emblem allows 255 levels per factor: this leaves room for Unknown, N/A, OTHER
    tidy_bins=200,              # numeric factors with more distinct values than max_levels: ~this many fine bins
    tail=0.001,                 # ... with the outer 0.1% of exposure at each end capped / floored
    min_expo=0.0,               # categorical levels under this share of exposure -> OTHER (0 = keep every level)
    point_min=0.02,             # spikes: a value with at least 2% of exposure ...
    point_ratio=3,              # ... and 3x its neighbours' average is always its own level, never merged
    raw_max_levels=1000,        # raw view: numeric factors with up to this many distinct values show every value
    raw_bins=250,               # ... beyond that, this many fine bins
    raw_max_cat_levels=1000,    # raw view: categorical factors show up to this many levels
)
TIDY_OVERRIDES = {              # replace a suggestion: {"edges": [...]} or {"levels": [...]}
    # "sum_insured_buildings": {"edges": [100000, 200000, 300000, 500000, 750000, 1000000, 2000000]},
}
BASE_LEVELS = {                 # base level per factor (a level name as shown in the report)
    # "employees": "5-9",
}

# ---- indicative Emblem grouping, per peril ----
MIN_CLAIMS_GROUP = {"default": 100, "ALL": 200, "Fire": 50, "Storm": 50, "Flood": 25}
MAX_GROUPS = 15

# ---- feature importance and correlation ----
IMPORTANCE = dict(
    folds=5,                    # k-fold cross-validation: each fifth of the data is held out once
    perms=2,                    # shuffles per factor per fold (averaged)
    max_trees=1000, learning_rate=0.05, leaves=15,   # gradient boosting model (stops early, usually far sooner)
    early_stop=30,              # stop when 30 trees in a row don't improve the validation deviance
    min_claims_leaf=10,         # each leaf needs enough rows for ~10 claims on average (limits over-fitting)
    power=1.5,                  # Tweedie power
    max_rows=500_000,           # sample down to this many rows for speed (None = use all)
    seed=1,
    min_claims_warn=300,        # warn when a peril has fewer claims than this
    context=True,               # include product and the cover flags as features
)
CORRELATION = dict(flag=0.3, strong=0.7)        # list pairs with Cramér's V above flag; strong = likely aliasing

RADAR_NAMES = {}                # Radar variable names if different, e.g. {"product": "Product"}

# ====================================================================================

CFG = dict(EXPO=EXPO, PERIOD=PERIOD, PRODUCT=PRODUCT, PRODUCTS=PRODUCTS, SECTIONS=SECTIONS, FACTORS=FACTORS,
           CATEGORICAL=CATEGORICAL, APPLIES_TO=APPLIES_TO, JUNK=JUNK, STRING_PLACEHOLDERS=STRING_PLACEHOLDERS,
           MAPPINGS=MAPPINGS, RADAR_NAMES=RADAR_NAMES, BASE_LEVELS=BASE_LEVELS, VERSION=VERSION,
           SOURCE=SAS_FILE, SOURCE_NAME=Path(SAS_FILE).name,
           PERIL_COLS={p: (COUNT_COL.format(peril=p), COST_COL.format(peril=p)) for p in PERILS})
S = dict(TIDY, min_claims_group=MIN_CLAIMS_GROUP, max_groups=MAX_GROUPS)
SHOW = {k: bool(v) for k, v in REPORT_SECTIONS.items()}
if not PERIOD:
    SHOW["stability"] = False


def load():
    perils = CFG["PERIL_COLS"]
    counts = [c for c, _ in perils.values()]
    costs = [k for _, k in perils.values()]
    cols = FACTORS + [EXPO, PRODUCT] + ([PERIOD] if PERIOD else []) + list(SECTIONS.values()) + counts + costs
    print(f"reading {SAS_FILE} ...")
    df = load_data(SAS_FILE, cols)
    df, notes = prepare(df, EXPO, counts, costs)
    for c in CATEGORICAL:                                  # SAS numeric codes 101.0 -> "101"
        s = df[c]
        if s.dtype.kind in "fi" and (s.dropna() % 1 == 0).all():
            df[c] = s.astype("Int64").astype("string")
        else:
            df[c] = s.astype("string").str.strip()
    df[PRODUCT] = df[PRODUCT].astype("string").str.strip()
    return df, perils, notes


def run_importance(eng, cat, tidy_specs, reserved, perils, which):
    """Gradient boosting importance for each peril in `which` ('ALL' = total cost)."""
    from importance import gbm_importance
    X = pd.DataFrame(index=eng.index)
    for f in FACTORS:                                       # categoricals as their modelling levels
        X[f] = level_series(eng[f], tidy_specs[f])[0] if cat[f] else eng[f]
    context = ([PRODUCT] + list(SECTIONS.values())) if IMPORTANCE["context"] else []
    for c in context:
        X[c] = eng[c].astype("string")
    X[EXPO] = eng[EXPO]
    out = {}
    for p in which:
        cc, kc = ("_n_all", "_c_all") if p == "ALL" else perils[p]
        X["_n"], X["_c"] = eng[cc], eng[kc]
        t0 = time.time()
        out[p] = gbm_importance(X, FACTORS, [f for f in FACTORS if cat[f]], "_n", "_c", EXPO, IMPORTANCE, context,
                                {f: ("N/A" if cat[f] else reserved[f]) for f in FACTORS if f in reserved})
        print(f"  importance {p}: {time.time() - t0:.0f}s")
    return out


def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    raw, perils, notes = load()
    eng, log, reserved, found = engineer(raw, CFG)
    eng["_n_all"] = sum(eng[c] for c, _ in perils.values())
    eng["_c_all"] = sum(eng[k] for _, k in perils.values())
    colours = _colours(PRODUCTS)

    print("checking factors ...")
    checks = {f: factor_checks(raw, eng, f, CFG, reserved) for f in FACTORS}
    cat = {f: (f in CATEGORICAL or not pd.api.types.is_numeric_dtype(eng[f])) for f in FACTORS}
    raw_specs = {f: raw_spec(eng, f, cat[f], reserved.get(f), EXPO, S) for f in FACTORS}
    tidy_specs = {f: tidy_spec(eng, f, cat[f], reserved.get(f), EXPO, "_n_all", "_c_all", S, TIDY_OVERRIDES.get(f))
                  for f in FACTORS}

    # reserved codes for Radar: N/A from engineer(); junk code one below it (numeric factors with JUNK rules)
    junk_codes = {}
    for f in JUNK:
        if f in cat and not cat[f]:
            na = reserved.get(f)
            junk_codes[f] = (na - 1) if na is not None else reserved_value(eng.loc[_real(eng, f, None), f])
    formulas = {f: radar_formula(f, CFG, reserved.get(f), junk_codes.get(f)) for f in FACTORS}

    report_perils = (["ALL"] if OUTPUTS["all_report"] else []) + (PERILS if OUTPUTS["peril_reports"] else [])
    imp = {}
    if SHOW["importance"] and report_perils:
        print("feature importance ...")
        need = ["ALL"] + PERILS if OUTPUTS["all_report"] else PERILS
        imp = run_importance(eng, cat, tidy_specs, reserved, perils, need)

    tidy_tables, files, group_labels = {}, {}, {}
    for peril in ["ALL"] + (PERILS if OUTPUTS["peril_reports"] else []):
        building = peril in report_perils
        print(f"{'building report' if building else 'preparing'}: {peril} ...")
        scope = "all perils" if peril == "ALL" else peril
        rep = HtmlReport(title=f"Pre-modelling review {VERSION} — {scope}",
                         subtitle=f"{Path(SAS_FILE).name} — raw data, suggested Radar levels and indicative Emblem "
                                  f"groups ({'total' if peril == 'ALL' else peril} burning cost)")
        cc, kc = ("_n_all", "_c_all") if peril == "ALL" else perils[peril]
        rep.meta = {"Source": SAS_FILE, "Rows": f"{len(eng):,}", "Exposure": f"{eng[EXPO].sum():,.0f}",
                    "Claims": f"{eng[cc].sum():,.0f}", "Claim cost": f"{eng[kc].sum():,.0f}",
                    "Min claims per group": f"{S['min_claims_group'].get(peril, S['min_claims_group']['default']):,}"}
        rep.checks = notes if SHOW["data_checks"] else []
        rows, sections = [], {}
        for f in FACTORS:
            v = factor_views(eng, f, cat[f], tidy_specs[f], raw_specs[f], peril, CFG, S)
            if building and add_factor_section(rep, eng, f, cat[f], checks[f], log, raw_specs[f], tidy_specs[f], v,
                                               peril, CFG, colours, S, SHOW, formulas[f]):
                sections[f] = rep.sections.pop()
            c = checks[f]
            flags = c["flags"] + factor_flags(tidy_specs[f], v["tabs"]["tidy"][0])
            row = {"factor": f, "type": c["type"], "missing (raw)": c["missing_raw"], "N/A": c["na_share"],
                   "Unknown": c["unknown"], "actions": sum(1 for lv, _ in flags if lv == "action"),
                   "checks": sum(1 for lv, _ in flags if lv == "check"),
                   "bands": len(v["tabs"]["tidy"][0]), "groups": len(v["tabs"]["group"][0]),
                   "spread": v["spread"], "base": v["base"] or "average"}
            if peril in imp:
                t = imp[peril][0]
                row["importance"] = t.loc[f, "share"]
                ranked = [i for i in t.index if not t.loc[i, "context"] and i != "_noise_"]
                row["imp_rank"] = f"{ranked.index(f) + 1}" + (" (noise)" if t.loc[f, "below_noise"] else "")
            rows.append(row)
            if peril == "ALL":
                tt = v["tabs"]["tidy"][0]
                tidy_tables[f] = {"table": tt, "base": v["tidy_base"],
                                  "xvals": {} if cat[f] else variate_x(eng, f, v["tabs"]["tidy"][1], EXPO)}
                group_labels[f] = v["tabs"]["tidy" if cat[f] else "group"][1]   # for correlation
        if not building:
            continue
        ov = pd.DataFrame(rows).set_index("factor")
        order = FACTORS if KEEP_ORDER_AS_LISTED else list(ov.sort_values("spread", ascending=False).index)
        rep.sections = [sections[f] for f in order if f in sections]
        imp_cols = [c for c in ("importance", "imp_rank") if c in ov.columns]
        parts = []
        if peril == "ALL":
            if OUTPUTS["peril_reports"]:
                parts.append("<p class='note'>Peril reports: " + " · ".join(
                    f'<a href="review_{p}_{VERSION}.html">{_esc(p)}</a>' for p in PERILS) + "</p>")
            if SHOW["data_checks"]:
                prod_tab, unexpected, by_year = product_summary(raw, CFG, perils)
                parts.append(products_html(CFG, prod_tab, unexpected, by_year,
                                           section_mix(raw, CFG) if SHOW["cover_check"] else None))
            if SHOW["cover_check"]:
                parts.append(cover_html(*cover_check(raw, CFG, perils)))
            if SHOW["factor_table"]:
                cols = ["type", "missing (raw)", "N/A", "Unknown", "actions", "checks", "bands", "groups", "spread",
                        "base"] + imp_cols
                parts.append(factor_table_html(ov[cols].sort_values(["actions", "checks"], ascending=False), True))
        elif SHOW["factor_table"]:
            cols = ["bands", "groups", "spread", "base"] + imp_cols + ["actions", "checks"]
            parts.append(factor_table_html(ov[cols].sort_values("importance" if imp_cols else "spread",
                                                                ascending=False)))
        if peril in imp:
            from importance import importance_cross_table, importance_html, plot_importance
            t, info = imp[peril]
            fig = plot_importance(t, info, f"Feature importance — {scope}")
            html = importance_html(t, info, scope, HtmlReport._fig(fig))
            if peril == "ALL" and len(imp) > 1:
                html += importance_cross_table({p: imp[p] for p in imp}, FACTORS)
            parts.append(html)
        if peril == "ALL" and SHOW["correlation"]:
            from importance import correlation_html, correlation_matrix, plot_correlation
            labels = dict(group_labels)
            context = [PRODUCT] + list(SECTIONS.values())
            for c in context:
                labels[c] = eng[c].astype("string").fillna("Unknown")
            m = correlation_matrix(labels, eng[EXPO].to_numpy(float))
            parts.append(correlation_html(m, HtmlReport._fig(plot_correlation(m, "Cramér's V between factors")),
                                          context, CORRELATION["flag"], CORRELATION["strong"]))
        rep.summary_html = "".join(parts)
        files[peril] = rep.save(out / f"review_{peril}_{VERSION}.html")

    for p, f in files.items():
        print(f"{p + ':':<8} {f}")
    if OUTPUTS["radar_spec"]:
        spec = write_spec(out / f"radar_spec_{VERSION}.xlsx", out, CFG, FACTORS, list(checks.values()), tidy_specs,
                          reserved, junk_codes, found, raw, eng, tidy_tables, VERSION)
        print(f"spec:    {spec}")
    return eng, tidy_specs


if __name__ == "__main__":
    ENGINEERED, TIDY_SPECS = main()
