"""
Pre-modelling review — data checks, raw data, suggested Radar bands and indicative Emblem groups.
Replaces run_readiness.py and run_banding.py.

How to run (banding.py, readiness.py and review.py must be in the same folder):
    Command prompt:   python run_review.py
    Jupyter:          %run run_review.py

One-off setup:
    pip install pandas plotly pyreadstat openpyxl

Outputs (in OUT_DIR, all tagged with VERSION):
    review_ALL_v1.html       data checks + every factor on all perils combined
    review_<PERIL>_v1.html   every factor for one peril (same layout)
    radar_spec_v1.xlsx       Radar set-up: product rules, formulas, banding rules, level order, base levels
    tidy_map_<factor>_v1.csv long categorical mappings to join in Radar instead of typing rules

Each factor shows three charts (exposure bars stacked by product, relativity lines):
    1. Raw data   2. Suggested bands for modelling (Radar)   3. Indicative Emblem grouping

Workflow:
    1. Edit CONFIG, run, open review_ALL. Resolve the "actions" by pasting the proposed lines into
       APPLIES_TO / JUNK and running again.
    2. Check the suggested bands in each peril's report; change any with TIDY_OVERRIDES.
    3. Set up Radar from radar_spec. Final grouping happens in Emblem.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from banding import HtmlReport, load_data, prepare
from readiness import (_colours, cover_check, engineer, factor_checks, product_summary, reserved_value,
                       section_mix)
from review import (SPECIAL, add_factor_section, all_summary_html, factor_table_html, factor_views, radar_formula,
                    raw_spec, tidy_spec, variate_x, write_spec, _real, other_share)

# =============================================================================== CONFIG

SAS_FILE = r"Z:\Pricing\SME\Modelling\sme_property_2026.sas7bdat"
OUT_DIR = r"Z:\Pricing\SME\Modelling\review"
VERSION = "v1"

EXPO = "exposure"
PERIOD = "policy_year"
SHOW_STABILITY = True

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

# ---- suggested modelling bands (Radar) - the same for every peril ----
TIDY = dict(
    tidy_bins=30,               # target number of round-number bands before merging
    values_max=30,              # whole-number factors with up to this many values keep one band per value
    tail=0.005,                 # cap / floor where less than this share of exposure lies beyond
    min_expo=0.005,             # merge bands / levels under this share of exposure ...
    min_claims_tidy=10,         # ... or with fewer claims (all perils) — levels go to OTHER
    raw_max_levels=300,         # raw view: numeric factors with up to this many distinct values show every value
    raw_bins=100,               # ... beyond that, this many fine equal-exposure bins (categoricals are never binned)
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

RADAR_NAMES = {}                # Radar variable names if different, e.g. {"product": "Product"}

# ====================================================================================

CFG = dict(EXPO=EXPO, PERIOD=PERIOD, PRODUCT=PRODUCT, PRODUCTS=PRODUCTS, SECTIONS=SECTIONS, FACTORS=FACTORS,
           CATEGORICAL=CATEGORICAL, APPLIES_TO=APPLIES_TO, JUNK=JUNK, STRING_PLACEHOLDERS=STRING_PLACEHOLDERS,
           MAPPINGS=MAPPINGS, RADAR_NAMES=RADAR_NAMES, BASE_LEVELS=BASE_LEVELS, VERSION=VERSION,
           SOURCE=SAS_FILE, SOURCE_NAME=Path(SAS_FILE).name,
           PERIL_COLS={p: (COUNT_COL.format(peril=p), COST_COL.format(peril=p)) for p in PERILS})
S = dict(TIDY, min_claims_group=MIN_CLAIMS_GROUP, max_groups=MAX_GROUPS, show_stability=SHOW_STABILITY)


def load():
    perils = CFG["PERIL_COLS"]
    counts = [c for c, _ in perils.values()]
    costs = [k for _, k in perils.values()]
    cols = FACTORS + [EXPO, PERIOD, PRODUCT] + list(SECTIONS.values()) + counts + costs
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
    raw_specs = {f: raw_spec(eng, f, cat[f], reserved.get(f), EXPO, TIDY["raw_max_levels"], TIDY["raw_bins"])
                 for f in FACTORS}
    tidy_specs = {f: tidy_spec(eng, f, cat[f], reserved.get(f), EXPO, "_n_all", S, TIDY_OVERRIDES.get(f))
                  for f in FACTORS}

    # reserved codes for Radar: N/A from engineer(); junk code one below it (numeric factors with JUNK rules)
    junk_codes = {}
    for f in JUNK:
        if not cat[f]:
            na = reserved.get(f)
            junk_codes[f] = (na - 1) if na is not None else reserved_value(eng.loc[_real(eng, f, None), f])
    formulas = {f: radar_formula(f, CFG, reserved.get(f), junk_codes.get(f)) for f in FACTORS}

    tidy_tables, files = {}, {}
    for peril in ["ALL"] + PERILS:
        print(f"building report: {peril} ...")
        scope = "all perils" if peril == "ALL" else peril
        rep = HtmlReport(title=f"Pre-modelling review {VERSION} — {scope}",
                         subtitle=f"{Path(SAS_FILE).name} — raw data, suggested Radar bands and indicative Emblem "
                                  f"groups ({'total' if peril == 'ALL' else peril} burning cost)")
        cc, kc = ("_n_all", "_c_all") if peril == "ALL" else perils[peril]
        rep.meta = {"Source": SAS_FILE, "Rows": f"{len(eng):,}", "Exposure": f"{eng[EXPO].sum():,.0f}",
                    "Claims": f"{eng[cc].sum():,.0f}", "Claim cost": f"{eng[kc].sum():,.0f}",
                    "Min claims per group": f"{S['min_claims_group'].get(peril, S['min_claims_group']['default']):,}"}
        rep.checks = notes
        rows, sections = [], {}
        for f in FACTORS:
            v = factor_views(eng, f, cat[f], tidy_specs[f], raw_specs[f], peril, CFG, S)
            add_factor_section(rep, eng, f, cat[f], checks[f], log, raw_specs[f], tidy_specs[f], v, peril, CFG,
                               colours, S, formulas[f])
            sections[f] = rep.sections.pop()
            c = checks[f]
            rows.append({"factor": f, "type": c["type"], "missing (raw)": c["missing_raw"], "N/A": c["na_share"],
                         "Unknown": c["unknown"],
                         "actions": sum(1 for lv, _ in c["flags"] if lv == "action")
                         + (1 if len(v["tabs"]["tidy"][0]) > 255 else 0)
                         + (1 if other_share(v["tabs"]["tidy"][0]) >= 0.2 else 0),
                         "checks": sum(1 for lv, _ in c["flags"] if lv == "check"),
                         "bands": len(v["tabs"]["tidy"][0]), "groups": len(v["tabs"]["group"][0]),
                         "spread": v["spread"], "base": v["base"] or "average"})
            if peril == "ALL":
                tt = v["tabs"]["tidy"][0]
                tidy_tables[f] = {"table": tt, "base": v["tidy_base"],
                                  "xvals": {} if cat[f] else variate_x(eng, f, v["tabs"]["tidy"][1], EXPO)}
        ov = pd.DataFrame(rows).set_index("factor")
        order = FACTORS if KEEP_ORDER_AS_LISTED else list(ov.sort_values("spread", ascending=False).index)
        rep.sections = [sections[f] for f in order]
        if peril == "ALL":
            prod_tab, unexpected, by_year = product_summary(raw, CFG, perils)
            cover_tab, cover_flags = cover_check(raw, CFG, perils)
            rep.summary_html = all_summary_html(CFG, prod_tab, unexpected, by_year, section_mix(raw, CFG),
                                                cover_tab, cover_flags,
                                                ov.sort_values(["actions", "checks"], ascending=False))
        else:
            rep.summary_html = factor_table_html(ov[["bands", "groups", "spread", "base", "actions", "checks"]]
                                                 .sort_values("spread", ascending=False))
        files[peril] = rep.save(out / f"review_{peril}_{VERSION}.html")

    spec = write_spec(out / f"radar_spec_{VERSION}.xlsx", out, CFG, FACTORS, list(checks.values()), tidy_specs,
                      reserved, junk_codes, found, raw, eng, tidy_tables, VERSION)
    for p, f in files.items():
        print(f"{p + ':':<8} {f}")
    print(f"spec:    {spec}")
    return eng, tidy_specs


if __name__ == "__main__":
    ENGINEERED, TIDY_SPECS = main()
