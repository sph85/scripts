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
    correlation_v1.csv       Cramér's V for every pair of factors (when correlation is on)
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

import difflib
import re
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

# ---- which rows to review, and what the exposure bars are split by ----
FILTER = {}                     # keep only these rows: {column: [values]}, e.g. {"product": ["RE"]};
                                # several columns must all match. {} = all rows
SPLIT_BY = None                 # column the exposure bars are stacked by, e.g. "property_type" (None = PRODUCT)
SPLIT_LEVELS = None             # its values in the order to show (None = the 8 largest; the rest show as "(other)")

PERILS = ["AD", "Fire", "EoW", "Theft", "Storm", "Flood"]
COUNT_COL = "No_Of_Claims_{peril}"
COST_COL = "Claim_Cost_{peril}"

FACTORS = ["sum_insured_buildings", "sum_insured_contents", "turnover", "years_trading", "employees",
           "trade_code", "construction", "occupancy"]
CATEGORICAL = ["trade_code", "construction", "occupancy"]
NO_COMMAS = []                  # numeric factors shown without thousands separators (years are detected automatically)
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
    min_claims_fit=20,          # below this many claims a peril's model isn't fitted (noted in the report)
    context=True,               # include product and the cover flags as features
)
CORRELATION = dict(
    flag=0.3,                   # list pairs of factors with Cramér's V at or above this
    strong=0.7,                 # ... and call them "strong" (likely to alias in Emblem) from this
    heatmap_max=20,             # also show the full matrix (collapsed) when there are at most this many factors
)                               # every pair is always written to correlation_<VERSION>.csv

RADAR_NAMES = {}                # Radar variable names if different, e.g. {"product": "Product"}

# ====================================================================================

def build_config():
    """Collect the settings above into CFG / S / SHOW. Called once below; tests call it again after
    changing a setting (e.g. run_review.FACTORS = [...]) so the run picks the change up."""
    global CFG, S, SHOW
    CFG = dict(EXPO=EXPO, PERIOD=PERIOD, PRODUCT=PRODUCT, PRODUCTS=PRODUCTS, SECTIONS=SECTIONS, FACTORS=FACTORS,
               CATEGORICAL=CATEGORICAL, NO_COMMAS=NO_COMMAS, APPLIES_TO=APPLIES_TO, JUNK=JUNK,
               STRING_PLACEHOLDERS=STRING_PLACEHOLDERS, MAPPINGS=MAPPINGS, RADAR_NAMES=RADAR_NAMES,
               BASE_LEVELS=BASE_LEVELS, VERSION=VERSION, SOURCE=SAS_FILE, SOURCE_NAME=Path(SAS_FILE).name,
               POINT_MIN=TIDY["point_min"], POINT_RATIO=TIDY["point_ratio"],
               PERIL_COLS={p: (COUNT_COL.format(peril=p), COST_COL.format(peril=p)) for p in PERILS})
    S = dict(TIDY, min_claims_group=MIN_CLAIMS_GROUP, max_groups=MAX_GROUPS, no_commas=NO_COMMAS)
    SHOW = {k: bool(v) for k, v in REPORT_SECTIONS.items()}
    if not PERIOD:
        SHOW["stability"] = False


build_config()

def load():
    perils = CFG["PERIL_COLS"]
    counts = [c for c, _ in perils.values()]
    costs = [k for _, k in perils.values()]
    cols = FACTORS + [EXPO, PRODUCT] + ([PERIOD] if PERIOD else []) + list(SECTIONS.values()) + counts + costs
    cols += [c for c in list(FILTER) + ([SPLIT_BY] if SPLIT_BY else []) if c not in cols]
    print(f"reading {SAS_FILE} ...")
    df = load_data(SAS_FILE, cols)
    df, filter_note, filter_warns = apply_filter(df)
    df, notes = prepare(df, EXPO, counts, costs)
    notes = ([filter_note] if filter_note else []) + [f"Config: {w}" for w in filter_warns] + list(notes)
    # categoricals stored as numbers in the source (e.g. trade code 101): Radar reads them as numbers, so the
    # spec writes numeric rules for them
    CFG["NUMERIC_CODES"] = [c for c in CATEGORICAL if c in df and df[c].dtype.kind in "fi"]
    for c in CATEGORICAL:                                  # SAS numeric codes 101.0 -> "101"
        s = df[c]
        if s.dtype.kind in "fi" and (s.dropna() % 1 == 0).all():
            df[c] = s.astype("Int64").astype("string")
        else:
            df[c] = s.astype("string").str.strip()
    df[PRODUCT] = df[PRODUCT].astype("string").str.strip()
    return df, perils, notes


class ConfigError(Exception):
    """A setting that would silently give wrong results: the run stops before reading any data."""


def _close(name, options):
    opts = [str(o) for o in options]
    same = [o for o in opts if o.lower() == str(name).lower() and o != str(name)]      # Re -> RE
    m = same or difflib.get_close_matches(str(name), opts, n=1, cutoff=0.6)
    return f" (did you mean {m[0]!r}?)" if m else ""


def check_config():
    """
    Catch typos before anything runs. Anything that would change the data or the levels stops the run with a
    list of every problem; display-only slips are warnings (printed and listed under Data checks).
    """
    errors, warns = [], []
    fs = set(FACTORS)
    dup = sorted({f for f in FACTORS if FACTORS.count(f) > 1})
    if dup:
        errors.append(f"FACTORS lists {', '.join(dup)} more than once")
    for name, d in [("APPLIES_TO", APPLIES_TO), ("JUNK", JUNK), ("MAPPINGS", MAPPINGS),
                    ("TIDY_OVERRIDES", TIDY_OVERRIDES), ("BASE_LEVELS", BASE_LEVELS)]:
        for f in d:
            if f not in fs:
                errors.append(f"{name} names {f!r}, which isn't in FACTORS{_close(f, FACTORS)}")
    for name, lst in [("CATEGORICAL", CATEGORICAL), ("NO_COMMAS", NO_COMMAS)]:
        for f in lst:
            if f not in fs:
                warns.append(f"{name} names {f!r}, which isn't in FACTORS, so it's ignored{_close(f, FACTORS)}")
    for f, rule in APPLIES_TO.items():
        if not isinstance(rule, dict):
            errors.append(f"APPLIES_TO[{f!r}] should be like {{\"products\": [\"CPO\", \"RE\"]}}")
            continue
        bad = set(rule) - {"products", "section", "reserved"}
        if bad:
            errors.append(f"APPLIES_TO[{f!r}] has {', '.join(sorted(bad))}; use products, section and/or reserved"
                          + _close(sorted(bad)[0], ["products", "section", "reserved"]))
        if not {"products", "section"} & set(rule):
            errors.append(f"APPLIES_TO[{f!r}] needs products and/or section")
        prods = rule.get("products", [])
        if isinstance(prods, str):
            errors.append(f"APPLIES_TO[{f!r}] products should be a list, e.g. [{prods!r}]")
            prods = [prods]
        for p in prods:
            if p not in PRODUCTS:
                errors.append(f"APPLIES_TO[{f!r}]: product {p!r} isn't in PRODUCTS ({', '.join(PRODUCTS)}); "
                              f"names are case-sensitive{_close(p, PRODUCTS)}")
        if "section" in rule and rule["section"] not in SECTIONS:
            errors.append(f"APPLIES_TO[{f!r}]: section {rule['section']!r} isn't in SECTIONS "
                          f"({', '.join(SECTIONS)}){_close(rule['section'], SECTIONS)}")
    for f, rule in JUNK.items():
        if not isinstance(rule, dict):
            errors.append(f"JUNK[{f!r}] should be like {{\"values\": [9999]}}")
            continue
        bad = set(rule) - {"values", "below", "above"}
        if bad:
            errors.append(f"JUNK[{f!r}] has {', '.join(sorted(bad))}; use values, below and/or above"
                          + _close(sorted(bad)[0], ["values", "below", "above"]))
        if "values" in rule and not isinstance(rule["values"], (list, tuple, set)):
            errors.append(f"JUNK[{f!r}] values should be a list, e.g. [{rule['values']!r}]")
    for f, o in TIDY_OVERRIDES.items():
        if not isinstance(o, dict) or not {"edges", "levels"} & set(o):
            errors.append(f"TIDY_OVERRIDES[{f!r}] needs edges (numeric factors) or levels (categorical factors)")
        elif "edges" in o and f in CATEGORICAL:
            errors.append(f"TIDY_OVERRIDES[{f!r}] gives edges, but {f} is categorical: use levels")
    for name, d, ok in [("OUTPUTS", OUTPUTS, ["all_report", "peril_reports", "radar_spec"]),
                        ("REPORT_SECTIONS", REPORT_SECTIONS, ["data_checks", "cover_check", "factor_table",
                         "importance", "correlation", "factor_details", "raw_view", "tidy_view", "group_view",
                         "stability", "tables"])]:
        for k in d:
            if k not in ok:
                warns.append(f"{name} has {k!r}, which isn't a known switch, so it's ignored{_close(k, ok)}")
    if errors:
        raise ConfigError("CONFIG problems - nothing was run:\n" + "\n".join(f"  - {e}" for e in errors))
    for w in warns:
        print(f"CONFIG WARNING: {w}")
    return warns


def data_warnings(raw):
    """Settings that point at values the data doesn't have (checked once the data is read)."""
    warns = []
    prod = raw[PRODUCT].astype("string").str.strip()
    if PRODUCT not in FILTER:
        for p in PRODUCTS:
            if not (prod == p).fillna(False).any():
                warns.append(f"PRODUCTS lists {p!r} but no rows have it{_close(p, prod.dropna().unique())}")
    for f, rule in JUNK.items():
        if f in raw:
            s = raw[f]
            have = set(s.dropna().astype(str).str.strip()) | ({str(_tidyv(x)) for x in s.dropna().unique()}
                                                               if pd.api.types.is_numeric_dtype(s) else set())
            for v in rule.get("values", []):
                if str(_tidyv(v)) not in have and str(v) not in have:
                    warns.append(f"JUNK value {v!r} for {f} doesn't occur in the data")
    for f, o in TIDY_OVERRIDES.items():
        if "levels" in o and f in raw:
            have = set(raw[f].dropna().astype(str).str.strip())
            for lv in o["levels"]:
                if str(lv) not in have:
                    warns.append(f"TIDY_OVERRIDES level {lv!r} for {f} doesn't occur in the data{_close(lv, have)}")
    if SPLIT_LEVELS and (SPLIT_BY or PRODUCT) in raw:
        have = set(raw[SPLIT_BY or PRODUCT].dropna().astype(str).str.strip())
        for v in SPLIT_LEVELS:
            if str(v) not in have:
                warns.append(f"SPLIT_LEVELS value {v!r} doesn't occur in {SPLIT_BY or PRODUCT}{_close(v, have)}")
    for w in warns:
        print(f"CONFIG WARNING: {w}")
    return warns


def _tidyv(x):
    try:
        x = float(x)
        return int(x) if x.is_integer() else x
    except (TypeError, ValueError):
        return x


def filter_text():
    return "; ".join(f"{c} = {' or '.join(map(str, v if isinstance(v, (list, tuple, set)) else [v]))}"
                     for c, v in FILTER.items())


def filter_tag():
    """Added to every output name when FILTER is set, so filtered runs don't overwrite full ones: _RE."""
    vals = [str(x) for v in FILTER.values() for x in (v if isinstance(v, (list, tuple, set)) else [v])]
    return ("_" + "_".join(re.sub(r"[^A-Za-z0-9]+", "", x) for x in vals))[:40] if FILTER else ""


def apply_filter(df):
    """Keep only the rows matching FILTER (every listed column must hold one of its listed values)."""
    if not FILTER:
        return df, None, []
    keep = pd.Series(True, index=df.index)
    warns = []
    for col, vals in FILTER.items():
        vals = list(vals) if isinstance(vals, (list, tuple, set)) else [vals]
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            s, vals = s, [v for v in pd.to_numeric(pd.Series(vals), errors="coerce").dropna().tolist()]
        else:
            s, vals = s.astype("string").str.strip(), [str(v).strip() for v in vals]
        for v in vals:
            if not (s == v).fillna(False).any():
                warns.append(f"FILTER value {v!r} for {col} matches no rows{_close(str(v), s.dropna().astype(str).unique())}")
        keep &= s.isin(vals).fillna(False)
    if not keep.any():
        raise ConfigError(f"FILTER {FILTER} matches no rows")
    note = (f"Filter {filter_text()}: {int(keep.sum()):,} of {len(df):,} rows kept "
            f"({df.loc[keep, EXPO].sum() / df[EXPO].sum():.1%} of exposure)")
    print(note)
    for w in warns:
        print(f"CONFIG WARNING: {w}")
    return df.loc[keep].copy(), note, warns


def make_split(df):
    """The column the exposure bars are stacked by: its values in order, the rest as (other), blanks as (blank)."""
    col = SPLIT_BY or PRODUCT
    s = df[col]
    if s.dtype.kind in "fi" and (s.dropna() % 1 == 0).all():          # SAS codes 3.0 -> "3"
        s = s.astype("Int64")
    s = s.astype("string").str.strip().replace("", pd.NA).fillna("(blank)")
    present = set(s.unique())
    if SPLIT_LEVELS:
        levels = [str(x) for x in SPLIT_LEVELS]
    elif col == PRODUCT:
        levels = list(PRODUCTS)
    else:
        levels = list(df.groupby(s)[EXPO].sum().sort_values(ascending=False).index[:8])
    levels = [x for x in levels if x in present]
    lab = s.where(s.isin(levels) | (s == "(blank)"), "(other)")
    order = levels + [x for x in ("(other)", "(blank)") if x not in levels and (lab == x).any()]
    colours = _colours((list(PRODUCTS) if col == PRODUCT else []) + [x for x in order if not (col == PRODUCT and x in PRODUCTS)])
    colours.update({"(other)": "#a3a29c", "(blank)": "#c9c7bf"})
    return lab, order, col, colours


def run_importance(eng, cat, tidy_specs, reserved, perils, which, factors):
    """Gradient boosting importance for each peril in `which` ('ALL' = total cost)."""
    from importance import gbm_importance
    X = pd.DataFrame(index=eng.index)
    for f in factors:                                       # categoricals as their modelling levels
        X[f] = level_series(eng[f], tidy_specs[f])[0] if cat[f] else eng[f]
    context = ([PRODUCT] + list(SECTIONS.values())) if IMPORTANCE["context"] else []
    for c in context:
        X[c] = eng[c].astype("string")
    X[EXPO] = eng[EXPO]
    out, skipped = {}, {}
    for p in which:
        cc, kc = ("_n_all", "_c_all") if p == "ALL" else perils[p]
        X["_n"], X["_c"] = eng[cc], eng[kc]
        n_claims, cost = float(X["_n"].sum()), float(X["_c"].sum())
        if n_claims < IMPORTANCE.get("min_claims_fit", 20) or cost <= 0:    # e.g. AD with FILTER = RPO only
            skipped[p] = (f"only {n_claims:,.0f} claims in this data, too few to fit a model"
                          if cost > 0 else "no claim cost in this data")
            print(f"  importance {p}: not estimated - {skipped[p]}")
            continue
        t0 = time.time()
        try:
            out[p] = gbm_importance(X, factors, [f for f in factors if cat[f]], "_n", "_c", EXPO, IMPORTANCE,
                                    context, {f: ("N/A" if cat[f] else reserved[f]) for f in factors if f in reserved})
            print(f"  importance {p}: {time.time() - t0:.0f}s")
        except Exception as e:                         # one peril's model must never stop the whole run
            skipped[p] = f"the model couldn't be fitted ({type(e).__name__}: {str(e)[:120]})"
            print(f"  importance {p}: not estimated - {skipped[p]}")
    return out, skipped


def main():
    cfg_warns = check_config()                     # stops here, before reading any data, if a setting is wrong
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    raw, perils, notes = load()
    notes = [f"Config: {w}" for w in cfg_warns + data_warnings(raw)] + list(notes)
    eng, log, reserved, found = engineer(raw, CFG)
    eng["_n_all"] = sum(eng[c] for c, _ in perils.values())
    eng["_c_all"] = sum(eng[k] for _, k in perils.values())
    split_lab, split_levels, split_name, colours = make_split(eng)
    eng["_split"] = pd.Categorical(split_lab, categories=split_levels)
    CFG.update(SPLIT="_split", SPLIT_LEVELS=split_levels, SPLIT_NAME=split_name, FILTER_TEXT=filter_text())
    vtag = VERSION + filter_tag()

    # a factor with no values where it applies (e.g. years trading with FILTER = RPO only) is left out
    factors = [f for f in FACTORS if _real(eng, f, reserved.get(f)).any()]
    for f in FACTORS:
        if f not in factors:
            notes = list(notes) + [f"{f}: no values where it applies in this data — left out of this run"]

    print("checking factors ...")
    checks = {f: factor_checks(raw, eng, f, CFG, reserved) for f in factors}
    cat = {f: (f in CATEGORICAL or not pd.api.types.is_numeric_dtype(eng[f])) for f in factors}
    raw_specs = {f: raw_spec(eng, f, cat[f], reserved.get(f), EXPO, S) for f in factors}
    tidy_specs = {f: tidy_spec(eng, f, cat[f], reserved.get(f), EXPO, "_n_all", "_c_all", S, TIDY_OVERRIDES.get(f))
                  for f in factors}

    # reserved codes for Radar: N/A from engineer(); junk code one below it (numeric factors with JUNK rules)
    junk_codes = {}
    for f in JUNK:
        if f in cat and not cat[f]:
            na = reserved.get(f)
            junk_codes[f] = (na - 1) if na is not None else reserved_value(eng.loc[_real(eng, f, None), f])
    # N/A code as Radar sees it: a number for numeric fields, including categoricals stored as numbers
    radar_na = {}
    for f, r in reserved.items():
        if f in factors and r == "N/A" and f in CFG["NUMERIC_CODES"]:
            radar_na[f] = reserved_value(pd.to_numeric(raw[f], errors="coerce"), APPLIES_TO[f].get("reserved"))
        elif f in factors:
            radar_na[f] = r
    formulas = {f: radar_formula(f, CFG, radar_na.get(f), junk_codes.get(f)) for f in factors}

    report_perils = (["ALL"] if OUTPUTS["all_report"] else []) + (PERILS if OUTPUTS["peril_reports"] else [])
    imp, imp_skipped = {}, {}
    if SHOW["importance"] and report_perils:
        print("feature importance ...")
        need = ["ALL"] + PERILS if OUTPUTS["all_report"] else PERILS
        imp, imp_skipped = run_importance(eng, cat, tidy_specs, reserved, perils, need, factors)

    tidy_tables, files, group_labels = {}, {}, {}
    cache = {}                                     # raw and suggested levels, worked out once per factor
    for peril in ["ALL"] + (PERILS if OUTPUTS["peril_reports"] else []):
        building = peril in report_perils
        print(f"{'building report' if building else 'preparing'}: {peril} ...")
        scope = "all perils" if peril == "ALL" else peril
        rep = HtmlReport(title=f"Pre-modelling review {VERSION} — {scope}" + (f" — {filter_text()}" if FILTER else ""),
                         subtitle=f"{Path(SAS_FILE).name} — raw data, suggested Radar levels and indicative Emblem "
                                  f"groups ({'total' if peril == 'ALL' else peril} burning cost)")
        cc, kc = ("_n_all", "_c_all") if peril == "ALL" else perils[peril]
        rep.meta = {"Source": SAS_FILE, "Rows": f"{len(eng):,}", "Exposure": f"{eng[EXPO].sum():,.0f}",
                    "Claims": f"{eng[cc].sum():,.0f}", "Claim cost": f"{eng[kc].sum():,.0f}",
                    "Min claims per group": f"{S['min_claims_group'].get(peril, S['min_claims_group']['default']):,}",
                    **({"Filter": filter_text()} if FILTER else {}), "Bars split by": split_name}
        rep.checks = notes if SHOW["data_checks"] else []
        rows, sections = [], {}
        for f in factors:
            v = factor_views(eng, f, cat[f], tidy_specs[f], raw_specs[f], peril, CFG, S, cache)
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
        order = factors if KEEP_ORDER_AS_LISTED else list(ov.sort_values("spread", ascending=False).index)
        rep.sections = [sections[f] for f in order if f in sections]
        imp_cols = [c for c in ("importance", "imp_rank") if c in ov.columns]
        parts = []
        if peril == "ALL":
            if OUTPUTS["peril_reports"]:
                parts.append("<p class='note'>Peril reports: " + " · ".join(
                    f'<a href="review_{p}_{vtag}.html">{_esc(p)}</a>' for p in PERILS) + "</p>")
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
                html += importance_cross_table({p: imp[p] for p in imp}, factors)
            if peril == "ALL" and imp_skipped:
                html += ("<p class='note'>Not estimated: " + "; ".join(
                    f"{_esc(p)} ({_esc(why)})" for p, why in imp_skipped.items()) + ".</p>")
            parts.append(html)
        elif peril in imp_skipped:
            parts.append(f"<h2>Feature importance ({_esc(scope)})</h2><p class='note'>Not estimated: "
                         f"{_esc(imp_skipped[peril])}.</p>")
        if peril == "ALL" and SHOW["correlation"]:
            from importance import correlation_html, correlation_matrix, plot_correlation
            print("correlation ...")
            labels = dict(group_labels)
            context = [PRODUCT] + list(SECTIONS.values())
            for c in context:
                labels[c] = eng[c].astype("string").fillna("Unknown")
            m = correlation_matrix(labels, eng[EXPO].to_numpy(float))
            csv = out / f"correlation_{vtag}.csv"
            m.round(3).to_csv(csv, index_label="factor")
            imp_all = None
            if "ALL" in imp:
                t = imp["ALL"][0]
                imp_all = {f: (float(t.loc[f, "share"]), bool(t.loc[f, "below_noise"])) for f in factors}
            fig = (HtmlReport._fig(plot_correlation(m, "Cramér's V between factors"))
                   if len(factors) <= CORRELATION["heatmap_max"] else None)
            parts.append(correlation_html(m, context, CORRELATION["flag"], CORRELATION["strong"], imp_all, fig,
                                          csv.name))
        rep.summary_html = "".join(parts)
        files[peril] = rep.save(out / f"review_{peril}_{vtag}.html")

    for p, f in files.items():
        print(f"{p + ':':<8} {f}")
    if OUTPUTS["radar_spec"]:
        spec = write_spec(out / f"radar_spec_{vtag}.xlsx", out, CFG, factors, list(checks.values()), tidy_specs,
                          radar_na, junk_codes, found, raw, eng, tidy_tables, vtag)
        print(f"spec:    {spec}")
    return eng, tidy_specs


if __name__ == "__main__":
    try:
        ENGINEERED, TIDY_SPECS = main()
    except ConfigError as e:                       # a plain list of what to fix, not a traceback
        print(e)
        raise SystemExit(1)
