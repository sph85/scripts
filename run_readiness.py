"""
Step 1 — data readiness and feature engineering.

How to run (banding.py and readiness.py must be in the same folder):
    Command prompt:   python run_readiness.py
    Jupyter:          %run run_readiness.py

One-off setup:
    pip install pandas plotly pyreadstat openpyxl

Outputs (in OUT_DIR):
    readiness_report_v1.html    products, section mix, peril cover check, every factor's checks and charts
    readiness_spec_v1.xlsx      what to set up in Radar: product rules, N/A formulas, junk rules, mappings

Workflow:
    1. Edit CONFIG and run. Open the report: the Factors table lists "actions" first.
    2. For each action you agree with, paste the proposed line into APPLIES_TO or JUNK and run again.
       Repeat until the actions are resolved. The spec workbook then lists the Radar set-up.
    3. engineer() gives the same engineered data in Python for screening (step 2).
"""

from pathlib import Path

from banding import load_data, prepare
from readiness import (build_report, cover_check, engineer, factor_checks, product_summary, section_mix,
                       write_spec)

# =============================================================================== CONFIG

SAS_FILE = r"Z:\Pricing\SME\Modelling\sme_property_2026.sas7bdat"
OUT_DIR = r"Z:\Pricing\SME\Modelling\readiness"
VERSION = "v1"

EXPO = "exposure"
PERIOD = "policy_year"

PRODUCT = "product"
PRODUCTS = ["RPO", "CPO", "RE"]          # anything else (or blank) is reported and falls to UNKNOWN in Radar

SECTIONS = {                              # Y/N cover flags
    "Buildings": "Buildings_Cover",
    "Contents": "Contents_Cover",
}

PERILS = ["AD", "Fire", "EoW", "Theft", "Storm", "Flood"]
COUNT_COL = "No_Of_Claims_{peril}"
COST_COL = "Claim_Cost_{peril}"

# The wide list of candidate factors (raw, unbanded)
FACTORS = ["sum_insured_buildings", "sum_insured_contents", "turnover", "years_trading", "employees",
           "trade_code", "construction", "occupancy"]
CATEGORICAL = ["trade_code", "construction", "occupancy"]

# ---- engineering you've confirmed (paste proposed lines from the report) ----

# Not applicable: the factor only exists for these products and/or where this section is Y
APPLIES_TO = {
    # "years_trading": {"products": ["CPO", "RE"]},
    # "sum_insured_buildings": {"section": "Buildings"},
}

# Junk -> missing (Unknown). "values": placeholders; "below"/"above": valid range
JUNK = {
    # "employees": {"values": [999, 9999]},
    # "years_trading": {"above": 150},
}

# Text values treated as missing in every text factor
STRING_PLACEHOLDERS = ["", "UNKNOWN", "UNK", "N/A", "NA", "NULL", "NONE", "?", "-", "TBC", "NOT KNOWN"]

# Harmonise fields recorded differently: {"factor": {"raw": "mapped"}} or {"factor": "path/to/mapping.csv"}
MAPPINGS = {}

# Radar variable names, if different from the data's column names (Radar names can't contain spaces)
RADAR_NAMES = {
    # "product": "Product", "Buildings_Cover": "BuildingsCover",
}

# ====================================================================================

CFG = dict(EXPO=EXPO, PERIOD=PERIOD, PRODUCT=PRODUCT, PRODUCTS=PRODUCTS, SECTIONS=SECTIONS, FACTORS=FACTORS,
           CATEGORICAL=CATEGORICAL, APPLIES_TO=APPLIES_TO, JUNK=JUNK, STRING_PLACEHOLDERS=STRING_PLACEHOLDERS,
           MAPPINGS=MAPPINGS, RADAR_NAMES=RADAR_NAMES, VERSION=VERSION, SOURCE=SAS_FILE,
           SOURCE_NAME=Path(SAS_FILE).name)


def load():
    perils = {p: (COUNT_COL.format(peril=p), COST_COL.format(peril=p)) for p in PERILS}
    counts = [c for c, _ in perils.values()]
    costs = [k for _, k in perils.values()]
    cols = FACTORS + [EXPO, PERIOD, PRODUCT] + list(SECTIONS.values()) + counts + costs
    print(f"reading {SAS_FILE} ...")
    df = load_data(SAS_FILE, cols)
    df, notes = prepare(df, EXPO, counts, costs)
    for c in CATEGORICAL:                              # SAS numeric codes 101.0 -> "101"
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

    print("checking factors ...")
    checks = [factor_checks(raw, eng, f, CFG, reserved) for f in FACTORS]
    prod_tab, unexpected, by_year = product_summary(raw, CFG, perils)
    mix = section_mix(raw, CFG)
    cover_tab, cover_flags = cover_check(raw, CFG, perils)

    rpt = build_report(raw, eng, CFG, perils, checks, log, reserved, prod_tab, unexpected, by_year, mix,
                       cover_tab, cover_flags, notes, out / f"readiness_report_{VERSION}.html")
    spec = write_spec(out / f"readiness_spec_{VERSION}.xlsx", CFG, checks, reserved, log, found, eng)
    n_act = sum(1 for c in checks for lv, _ in c["flags"] if lv == "action")
    print(f"report: {rpt}\nspec:   {spec}\n{n_act} actions to review")
    return eng, checks


if __name__ == "__main__":
    ENGINEERED, CHECKS = main()
