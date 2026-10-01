"""
Pre-Emblem banding review by peril — HTML reports plus the Emblem input file.

How to run (banding.py must be in the same folder as this script):
    Command prompt:   python run_banding.py
    Jupyter:          %run run_banding.py
    VS Code/Spyder:   open this file and press Run

One-off setup:
    pip install pandas plotly pyreadstat

Outputs (in OUT_DIR, all tagged with SPEC_VERSION):
    banding_overview_v1.html       peril totals, factor x peril summary, relativities and mix by peril
    banding_report_v1_<PERIL>.html one per peril: raw view, fine view, suggested bands, stability
    band_spec_v1.json              every peril's bands (the record of what the models used)
    band_labels_v1.csv             code -> label lookup per peril and factor (Emblem labels / Radar tables)
    emblem_input_v1.csv            data + one banded column per factor per peril, e.g. sum_insured_band_AD

Workflow:
    1. Edit CONFIG, run, open the overview then each peril's report.
    2. To change a peril's bands for a factor, copy the line from that peril's report into
       OVERRIDES under that peril, edit it, and run again. Bump SPEC_VERSION when you settle.
"""

from pathlib import Path

import pandas as pd

from banding import (HtmlReport, apply_spec, heat_table, label_table, load_data, peril_overview,
                     prepare, review_all, save_spec, validate)

# =============================================================================== CONFIG

SAS_FILE = r"Z:\Pricing\SME\Modelling\sme_property_2026.sas7bdat"      # input dataset
OUT_DIR = r"Z:\Pricing\SME\Modelling\banding"                          # where outputs go
SPEC_VERSION = "v1"

# Column names (case doesn't matter — SAS upper-case names are matched automatically)
EXPO = "exposure"
PERIOD = "policy_year"          # set to None if you don't have one
SHOW_STABILITY = True           # False hides the stability-by-year charts (PERIOD is still kept)

# Perils: one claim count and one (capped) claim cost column per peril, built from these patterns
PERILS = ["AD", "Fire", "EoW", "Theft", "Storm", "Flood"]
COUNT_COL = "No_Of_Claims_{peril}"
COST_COL = "Claim_Cost_{peril}"

FACTORS = ["sum_insured", "turnover", "years_trading", "employees", "trade_group"]
CATEGORICAL = ["trade_group"]   # treat as categories even if stored as numbers (e.g. trade codes)
KEEP_COLUMNS = []               # extra columns to carry into the Emblem file (e.g. "policy_id")

# Minimum claims per band — perils with fewer claims usually need a lower figure
MIN_CLAIMS = {"default": 100, "Fire": 50, "Storm": 50, "Flood": 25}

SETTINGS = dict(
    max_bands=15,               # cap on numeric bands
    n_candidates=20,            # fine-view grid before merging
    min_expo_share=0.01,        # categorical levels below this exposure share -> OTHER
    show_raw=True,              # raw view: every value, or RAW_BINS unrounded bins
    raw_bins=100,
    raw_max_levels=100,         # whole-number factors with up to this many values show every value
)

# Your own bands, replacing the suggestion. Copy the line from the peril's report and edit it.
# "ALL" applies to every peril unless that peril has its own entry for the factor.
#   Numeric:     "factor": {"edges": [...]}
#   Categorical: "factor": {"levels": [...]}            (levels not listed go to OTHER)
OVERRIDES = {
    "ALL": {
        # "employees": {"edges": [2, 3, 5, 8]},
    },
    # "AD": {
    #     "sum_insured": {"edges": [25000, 50000, 100000, 250000, 500000]},
    # },
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
    perils = {p: (COUNT_COL.format(peril=p), COST_COL.format(peril=p)) for p in PERILS}
    counts = [c for c, _ in perils.values()]
    costs = [k for _, k in perils.values()]

    cols = FACTORS + [EXPO] + counts + costs + ([PERIOD] if PERIOD else []) + KEEP_COLUMNS
    print(f"reading {SAS_FILE} ...")
    df = load_data(SAS_FILE, cols)
    df, notes = prepare(df, EXPO, counts, costs)
    for c in CATEGORICAL:                     # SAS stores numeric codes as 101.0 -> use "101"
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and (s.dropna() % 1 == 0).all():
            df[c] = s.astype("Int64").astype("string")
        else:
            df[c] = s.astype("string").str.strip()

    min_claims = {p: MIN_CLAIMS.get(p, MIN_CLAIMS["default"]) for p in PERILS}
    period = PERIOD if SHOW_STABILITY else None
    expo_total = df[EXPO].sum()
    base_meta = {"Source": SAS_FILE, "Rows": f"{len(df):,}", "Exposure": f"{expo_total:,.0f}"}

    # ------------------------------------------------------------ one review per peril
    bands, summaries, files = {}, {}, {}
    for p, (cc, kc) in perils.items():
        ov = {**OVERRIDES.get("ALL", {}), **OVERRIDES.get(p, {})}
        ov = {k: _full_spec(df, k, o) for k, o in ov.items()}
        rep = HtmlReport(title=f"Banding review {v} — {p}",
                         subtitle=f"{Path(SAS_FILE).name} — {p} burning cost ({kc} / {EXPO}), exposure-weighted")
        rep.meta = {**base_meta, "Claims": f"{df[cc].sum():,.0f}", "Claim cost": f"{df[kc].sum():,.0f}",
                    "Min claims per band": f"{min_claims[p]:,}", "Overrides": ", ".join(ov) or "none"}
        rep.checks = notes
        bands[p], summaries[p] = review_all(
            df, FACTORS, expo=EXPO, claims=cc, cost=kc, period=period, categorical=CATEGORICAL,
            overrides=ov, show=False, report=rep, label=p, min_claims=min_claims[p],
            override_hint=f'OVERRIDES["{p}"]', **SETTINGS)
        files[p] = rep.save(out / f"banding_report_{v}_{p}.html")

    # ------------------------------------------------------------ overview across perils
    tot = pd.DataFrame({p: {"claims": df[cc].sum(), "cost": df[kc].sum()} for p, (cc, kc) in perils.items()}).T
    tot["frequency"] = tot.claims / expo_total
    tot["severity"] = tot.cost / tot.claims.where(tot.claims > 0)
    tot["burning_cost"] = tot.cost / expo_total
    tot["cost_share"] = tot.cost / tot.cost.sum()
    tot["min_claims"] = [min_claims[p] for p in tot.index]
    tot = tot.sort_values("cost", ascending=False)
    spread = pd.DataFrame({p: summaries[p]["rel_spread"] for p in PERILS}).reindex(FACTORS)
    spread = spread.loc[spread.max(axis=1).sort_values(ascending=False).index, list(tot.index)]

    from banding import _table, _slug
    links = "".join(f'<li><a href="{Path(f).name}">{p}</a></li>' for p, f in files.items())
    ov_rep = HtmlReport(title=f"Banding overview {v}", subtitle=f"{Path(SAS_FILE).name} — all perils")
    ov_rep.meta = {**base_meta, "Claims": f"{tot.claims.sum():,.0f}", "Claim cost": f"{tot.cost.sum():,.0f}"}
    ov_rep.checks = notes
    ov_rep.summary_html = (
        "<h2>Perils</h2>"
        + _table(tot, {"claims": ",.0f", "cost": ",.0f", "frequency": ".4f", "severity": ",.0f",
                       "burning_cost": ",.2f", "cost_share": ".1%", "min_claims": ",.0f"},
                 {"claims": "Claims", "cost": "Claim cost", "frequency": "Frequency", "severity": "Severity",
                  "burning_cost": "Burning cost", "cost_share": "Cost share", "min_claims": "Min claims/band"},
                 index_name="Peril")
        + f"<p class='note'>Peril reports: </p><ul class='checks'>{links}</ul>"
        + "<h2>Relativity spread by factor and peril</h2>"
          "<p class='note'>Highest ÷ lowest burning cost relativity across each peril's bands that meet its "
          "minimum claims — a rough one-way guide to which factors matter for which peril. Darker = bigger "
          "spread. Click a factor for its peril comparison below.</p>"
        + heat_table(spread, "Factor", links={f: _slug(f) for f in spread.index}))
    peril_overview(df, list(spread.index), perils, expo=EXPO, categorical=CATEGORICAL,
                   n_candidates=SETTINGS["n_candidates"], min_claims=min_claims, report=ov_rep)
    ov_path = ov_rep.save(out / f"banding_overview_{v}.html")

    # ------------------------------------------------------------ band definitions
    save_spec(bands, out / f"band_spec_{v}.json", v)
    labels = pd.concat([label_table(n, s).assign(peril=p) for p, b in bands.items() for n, s in b.items()],
                       ignore_index=True)
    labels.to_csv(out / f"band_labels_{v}.csv", index=False)
    print(f"overview:    {ov_path}")
    for p, f in files.items():
        print(f"{p + ':':<12} {f}")

    if WRITE_EMBLEM_FILE:
        banded = df.copy()
        for p, (cc, _) in perils.items():
            banded = apply_spec(banded, bands[p], suffix=f"_band_{p}")
            validate(df, banded, bands[p], expo=EXPO, claims=cc, suffix=f"_band_{p}", min_claims=min_claims[p])
        banded.to_csv(out / f"emblem_input_{v}.csv", index=False)
        print(f"emblem file: {out / f'emblem_input_{v}.csv'}")

    return bands, summaries


if __name__ == "__main__":
    BANDS, SUMMARIES = main()
