"""
Pre-Emblem banding review — produces an HTML report plus the Emblem input file.

How to run (banding.py must be in the same folder as this script):
    Command prompt:   python run_banding.py
    Jupyter:          %run run_banding.py
    VS Code/Spyder:   open this file and press Run

One-off setup:
    pip install pandas plotly pyreadstat

Workflow:
    1. Edit the CONFIG section, run, open the HTML report.
    2. If you want different bands for a factor, copy the suggested line from the report
       into OVERRIDES, change it, and run again (bump SPEC_VERSION when you settle on a set).
"""

from pathlib import Path

import pandas as pd

from banding import (HtmlReport, apply_spec, label_table, load_data, prepare, review_all,
                     save_spec, validate)

# =============================================================================== CONFIG

SAS_FILE = r"Z:\Pricing\SME\Modelling\sme_property_2026.sas7bdat"      # input dataset
OUT_DIR = r"Z:\Pricing\SME\Modelling\banding"                          # where outputs go
SPEC_VERSION = "v1"                                                      # in every output file name

# Column names (case doesn't matter — SAS upper-case names are matched automatically)
EXPO = "exposure"
COUNT = "claim_count"
COST = "claim_cost_capped"      # capped amount for the burning cost / Tweedie response
PERIOD = "policy_year"          # set to None if you don't have one

FACTORS = ["sum_insured", "turnover", "years_trading", "employees", "trade_group"]
CATEGORICAL = ["trade_group"]   # treat as categories even if stored as numbers (e.g. trade codes)
KEEP_COLUMNS = []               # extra columns to carry into the Emblem file (e.g. "policy_id")

SETTINGS = dict(
    min_claims=100,             # minimum claims per band — tune to book size
    max_bands=15,               # cap on numeric bands
    n_candidates=20,            # fine-view grid before merging
    min_expo_share=0.01,        # categorical levels below this exposure share -> OTHER
)

# Your own bands, replacing the suggestion. Copy the line from the report and edit it.
# Numeric:     "factor": {"edges": [...]}
# Categorical: "factor": {"levels": [...]}            (levels not listed go to OTHER)
OVERRIDES = {
    # "sum_insured": {"edges": [25000, 50000, 100000, 250000, 500000]},
    # "trade_group": {"levels": ["A", "B", "C", "E"]},
}

WRITE_EMBLEM_FILE = True        # set False while you're still iterating on the bands

# ====================================================================================


def _full_spec(df, name, o):
    """Turn a short override ({"edges": [...]} / {"levels": [...]}) into a full band spec."""
    if "type" in o:
        return o
    if "edges" in o:
        s = df[name]
        whole = bool((s.dropna() % 1 == 0).all())
        return {"type": "numeric", "edges": sorted(o["edges"]), "right": False, "integer": whole,
                "data_min": int(s.min()) if whole else None}
    levels = [str(x) for x in o["levels"]]
    raw = df[name].dropna().astype(str).unique()
    mapping = {k: (k if k in levels else "OTHER") for k in raw}
    if any(v == "OTHER" for v in mapping.values()) and "OTHER" not in levels:
        levels = levels + ["OTHER"]
    return {"type": "categorical", "map": mapping, "levels": levels, "other": "OTHER"}


def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    v = SPEC_VERSION

    cols = FACTORS + [EXPO, COUNT, COST] + ([PERIOD] if PERIOD else []) + KEEP_COLUMNS
    print(f"reading {SAS_FILE} ...")
    df = load_data(SAS_FILE, cols)
    df, notes = prepare(df, EXPO, COUNT, COST)
    for c in CATEGORICAL:                     # SAS stores numeric codes as 101.0 -> use "101"
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and (s.dropna() % 1 == 0).all():
            df[c] = s.astype("Int64").astype("string")
        else:
            df[c] = s.astype("string").str.strip()

    report = HtmlReport(
        title=f"Banding review {v}",
        subtitle=f"{Path(SAS_FILE).name} — burning cost relativities, exposure-weighted",
    )
    report.meta = {
        "Source": SAS_FILE,
        "Rows": f"{len(df):,}",
        "Exposure": f"{df[EXPO].sum():,.0f}",
        "Claims": f"{df[COUNT].sum():,.0f}",
        "Claim cost": f"{df[COST].sum():,.0f}",
        "Min claims per band": f"{SETTINGS['min_claims']:,}",
        "Overrides": ", ".join(OVERRIDES) or "none",
    }
    report.checks = notes

    overrides = {k: _full_spec(df, k, o) for k, o in OVERRIDES.items()}
    bands, summary = review_all(df, FACTORS, expo=EXPO, claims=COUNT, cost=COST, period=PERIOD,
                                categorical=CATEGORICAL, overrides=overrides,
                                show=False, report=report, **SETTINGS)

    path = report.save(out / f"banding_report_{v}.html")
    save_spec(bands, out / f"band_spec_{v}.json", v)
    labels = pd.concat([label_table(n, s) for n, s in bands.items()], ignore_index=True)
    labels.to_csv(out / f"band_labels_{v}.csv", index=False)
    print(f"report:      {path}")
    print(f"band spec:   {out / f'band_spec_{v}.json'}")
    print(f"band labels: {out / f'band_labels_{v}.csv'}")

    if WRITE_EMBLEM_FILE:
        banded = apply_spec(df, bands)
        validate(df, banded, bands, expo=EXPO, claims=COUNT, min_claims=SETTINGS["min_claims"])
        banded.to_csv(out / f"emblem_input_{v}.csv", index=False)
        print(f"emblem file: {out / f'emblem_input_{v}.csv'}")

    return bands, summary


if __name__ == "__main__":
    BANDS, SUMMARY = main()
