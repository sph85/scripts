"""
Regression tests for the pre-modelling review.

Put the tests folder next to run_review.py (and the other four scripts), then from that folder run:

    python tests/run_tests.py                  run every check; compare with the saved baseline if there is one
    python tests/run_tests.py --save-baseline  save this run's results as the baseline to compare against later

Jupyter: %run tests/run_tests.py   (or %run tests/run_tests.py --save-baseline)

What it does (about a minute):
  1. Makes the synthetic test data if it isn't there yet (tests/make_test_data.py: every awkward case so far).
  2. Checks that config mistakes stop the run (or warn) with a clear message.
  3. Runs the review on the test data three ways: as is, filtered to RE with bars split by property type,
     and filtered to RPO. Outputs go to tests/output/ (not your real output folder).
  4. Checks the results: Radar's banding rules give exactly the Python levels on every row, spikes are kept,
     no duplicate OTHER, years have no commas, ratio cut points survive being pasted into Radar, trade code
     gets number rules, small Unknown levels are flagged, filter and split work.
  5. Baseline: compares the spec workbook, the correlation table and every factor's indicative groups for
     every peril with the saved baseline, and lists anything that changed.

Use it after any change to the scripts. A FAIL in steps 2-4 is a bug. A baseline difference is either
intended (a change you meant to make: save a new baseline) or a side effect to look into.
The settings for the tests are fixed below, so changing your own CONFIG never affects the tests.
"""

import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_review as rr                                   # noqa: E402
from banding import _fmt                                   # noqa: E402
from review import band_names, factor_views, level_series  # noqa: E402

DATA = HERE / "data" / "test_data.xpt"
OUT = HERE / "output"
BASELINE = HERE / "baseline"

BASE = dict(
    VERSION="test", EXPO="exposure", PERIOD="policy_year", PRODUCT="product", PRODUCTS=["RPO", "CPO", "RE"],
    SECTIONS={"Buildings": "Buildings_Cover", "Contents": "Contents_Cover"},
    PERILS=["AD", "Fire", "EoW", "Theft", "Storm", "Flood"],
    COUNT_COL="No_Of_Claims_{peril}", COST_COL="Claim_Cost_{peril}",
    FACTORS=["sum_insured_buildings", "sum_insured_contents", "turnover", "years_trading", "employees", "trade_code",
             "construction", "occupancy", "year_built", "claims_per_property", "claims_per_location"],
    CATEGORICAL=["trade_code", "construction", "occupancy"], NO_COMMAS=[], KEEP_ORDER_AS_LISTED=True,
    OUTPUTS=dict(all_report=True, peril_reports=False, radar_spec=True),
    REPORT_SECTIONS=dict(data_checks=True, cover_check=True, factor_table=True, importance=False, correlation=True,
                         factor_details=True, raw_view=True, tidy_view=True, group_view=True, stability=True,
                         tables=True),
    FILTER={}, SPLIT_BY=None, SPLIT_LEVELS=None,
    APPLIES_TO={"years_trading": {"products": ["CPO", "RE"]}, "sum_insured_buildings": {"section": "Buildings"},
                "sum_insured_contents": {"section": "Contents"}, "trade_code": {"products": ["CPO", "RE"]}},
    JUNK={"employees": {"values": [9999]}, "years_trading": {"values": [999]}},
    STRING_PLACEHOLDERS=["", "UNKNOWN", "UNK", "N/A", "NA", "NULL", "NONE", "?", "-", "TBC", "NOT KNOWN"],
    MAPPINGS={}, TIDY_OVERRIDES={}, BASE_LEVELS={}, RADAR_NAMES={},
    TIDY=dict(max_levels=250, tidy_bins=200, tail=0.001, min_expo=0.0, point_min=0.02, point_ratio=3,
              raw_max_levels=1000, raw_bins=250, raw_max_cat_levels=1000),
    MIN_CLAIMS_GROUP={"default": 100, "ALL": 200, "Fire": 50, "Storm": 50, "Flood": 25}, MAX_GROUPS=15,
    IMPORTANCE=dict(folds=5, perms=2, max_trees=1000, learning_rate=0.05, leaves=15, early_stop=30,
                    min_claims_leaf=10, power=1.5, max_rows=500_000, seed=1, min_claims_warn=300,
                    min_claims_fit=20, context=True),
    CORRELATION=dict(flag=0.3, strong=0.7, heatmap_max=20),
)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail and not ok else ""))


def configure(out_name, **over):
    for k, v in {**BASE, **over}.items():
        setattr(rr, k, copy.deepcopy(v))
    rr.SAS_FILE = str(DATA)
    rr.OUT_DIR = str(OUT / out_name)
    rr.build_config()


def quiet(fn, *a, **k):
    """Run fn without its progress printing."""
    import builtins
    keep = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        return fn(*a, **k)
    finally:
        builtins.print = keep


# =============================================================================== Radar simulation

def read_source():
    """The test data as Radar would read it, with the config's column names."""
    import pyreadstat
    df, _ = pyreadstat.read_xport(str(DATA))
    names = {c.lower(): c for c in BASE["FACTORS"] + [BASE["PRODUCT"]] + list(BASE["SECTIONS"].values())}
    return df.rename(columns={c: names.get(c.lower(), c) for c in df.columns})


def radar_levels(src, f, rules, formulas, applies, junk, sections, digits=None):
    """Apply the spec's Formula and Banding rules to the source data the way Radar would."""
    x = src[f].copy()
    numeric = rules["Rules type"].iloc[0] == "Float"
    row = formulas[formulas.Factor == f]
    na_code = row["N/A code"].iloc[0] if len(row) else ""
    junk_code = row["Junk code"].iloc[0] if len(row) else ""
    if numeric:
        x = pd.to_numeric(x, errors="coerce")
    else:
        x = x.astype("string").str.strip().replace("", pd.NA)
    if junk_code != "" and f in junk:                       # Formula component: junk -> its code
        rule = junk[f]
        hit = x.isin(rule.get("values", []))
        if "below" in rule:
            hit |= x < rule["below"]
        if "above" in rule:
            hit |= x > rule["above"]
        x = x.where(~hit.fillna(False), float(junk_code))
    if f in applies:                                        # Formula component: not applicable -> N/A code
        rule = applies[f]
        ok = pd.Series(True, index=src.index)
        if "products" in rule:
            ok &= src[BASE["PRODUCT"]].astype(str).str.strip().isin(rule["products"])
        if "section" in rule:
            ok &= src[sections[rule["section"]]].astype(str).str.strip().str.upper().eq("Y")
        x = x.where(ok, float(na_code) if numeric else "N/A")

    def bound(v):
        if v == "" or not numeric:
            return v
        v = float(v)
        return float(f"{v:.{digits}g}") if digits else v

    out = pd.Series(pd.NA, index=src.index, dtype="object")
    missing = x.isna()
    for _, r in rules.iterrows():
        op, lo, uop, up, name = r["Lower Operator"], bound(r["Lower Bound"]), r["Upper Operator"], \
            bound(r["Upper Bound"]), r["Level name"]
        free = out.isna()
        if op == "NoValue":
            m = missing
        elif op == "Default":
            m = ~missing
        elif op == "=":
            m = (x == (lo if numeric else str(lo))) if numeric else x.eq(str(lo))
        else:
            m = {"<": x < lo, ">": x > lo, ">=": x >= lo}[op]
            if uop == "<":
                m &= x < up
        out[free & m.fillna(False)] = name
    return out


# =============================================================================== the checks

def test_config():
    print("\nConfig check")
    configure("config", APPLIES_TO={**BASE["APPLIES_TO"], "years_trading": {"products": ["CPO", "Re"]}})
    try:
        rr.check_config()
        check("a misspelt product in APPLIES_TO stops the run", False, "check_config passed it")
    except rr.ConfigError as e:
        check("a misspelt product in APPLIES_TO stops the run", "did you mean 'RE'" in str(e), str(e))
    configure("config", TIDY_OVERRIDES={"turnvoer": {"edges": [10000]}})
    try:
        rr.check_config()
        check("a misspelt factor in TIDY_OVERRIDES stops the run", False, "check_config passed it")
    except rr.ConfigError as e:
        check("a misspelt factor in TIDY_OVERRIDES stops the run", "did you mean 'turnover'" in str(e), str(e))
    configure("config", APPLIES_TO={**BASE["APPLIES_TO"], "years_trading": {"product": ["CPO", "RE"]}})
    try:
        rr.check_config()
        check("a misspelt rule key (product for products) stops the run", False)
    except rr.ConfigError as e:
        check("a misspelt rule key (product for products) stops the run", "did you mean 'products'" in str(e), str(e))
    configure("config", JUNK={**BASE["JUNK"], "employees": {"values": [99999]}})
    raw = quiet(lambda: rr.load()[0])
    w = quiet(rr.data_warnings, raw)
    check("a JUNK value that isn't in the data gives a warning", any("99999" in x for x in w), str(w))


def test_default():
    print("\nFull run on the test data (importance on)")
    configure("default", REPORT_SECTIONS={**BASE["REPORT_SECTIONS"], "importance": True})
    eng, tidy = quiet(rr.main)
    out = OUT / "default"
    for f in ["review_ALL_test.html", "radar_spec_test.xlsx", "correlation_test.csv"]:
        check(f"writes {f}", (out / f).exists())
    spec = pd.ExcelFile(out / "radar_spec_test.xlsx")
    rules = pd.read_excel(spec, "Banding rules", keep_default_na=False)
    formulas = pd.read_excel(spec, "Formulas", keep_default_na=False)
    checks = pd.read_excel(spec, "Factor checks", keep_default_na=False)
    levels = pd.read_excel(spec, "Levels", keep_default_na=False)
    src = read_source()
    finite = ~np.isinf(pd.to_numeric(src["claims_per_property"], errors="coerce").fillna(0))  # SAS can't hold infinity

    for f in BASE["FACTORS"]:
        py = level_series(eng[f], tidy[f])[0].astype(str)
        rd = radar_levels(src, f, rules[rules.Factor == f], formulas, BASE["APPLIES_TO"], BASE["JUNK"],
                          BASE["SECTIONS"]).astype(str)
        same = (py.to_numpy() == rd.to_numpy()) | ~finite.to_numpy()
        bad = (~same).sum()
        check(f"Radar rules give the Python levels on every row: {f}", bad == 0,
              f"{bad} rows differ, e.g. Python {list(py[~same][:3])} vs Radar {list(rd[~same][:3])}")
    f = "claims_per_property"
    py = level_series(eng[f], tidy[f])[0].astype(str)
    rd6 = radar_levels(src, f, rules[rules.Factor == f], formulas, BASE["APPLIES_TO"], BASE["JUNK"], BASE["SECTIONS"],
                       digits=6).astype(str)
    check("ratio cut points still work when Radar keeps only 6 significant digits",
          ((py.to_numpy() == rd6.to_numpy()) | ~finite.to_numpy()).all())
    check("infinite ratios (divided by zero) become Unknown",
          (py[~finite] == "Unknown").all() and (~finite).any())

    for f in BASE["FACTORS"]:                                # one level per value: each value in its own level
        if tidy[f].get("per_value"):
            names = dict(zip([float(x) for x in tidy[f]["values"]], band_names(tidy[f])))
            vals = pd.to_numeric(eng[f], errors="coerce")
            lab = level_series(eng[f], tidy[f])[0].astype(str)
            real = ~lab.isin(["Unknown", "N/A"]) & vals.notna()
            own = vals[real].map(lambda x: names[float(x)])
            check(f"every value lands in the level named after it: {f}", (lab[real] == own).all())
    check("the test data has a decimal factor with one level per value",
          bool(tidy["claims_per_location"].get("per_value")) and not tidy["claims_per_location"]["integer"])
    f = "claims_per_location"
    py = level_series(eng[f], tidy[f])[0].astype(str)
    rd6 = radar_levels(src, f, rules[rules.Factor == f], formulas, BASE["APPLIES_TO"], BASE["JUNK"], BASE["SECTIONS"],
                       digits=6).astype(str)
    check("one-level-per-value decimal cut points still work with 6 significant digits",
          (py.to_numpy() == rd6.to_numpy()).all())
    check("no duplicate OTHER in the suggested levels list (occupancy)",
          len(tidy["occupancy"]["levels"]) == len(set(tidy["occupancy"]["levels"])))

    tl = lambda f: list(levels[levels.Factor == f]["Level name"].astype(str))        # noqa: E731
    check("spike kept: 0 years trading is its own level", "0" in tl("years_trading"))
    check("spike kept: turnover default 100,000 is its own level", "100,000" in tl("turnover"))
    check("spike kept: year built default 1900 is its own level", "1900" in tl("year_built"))
    check("years are written without commas", not any("," in x for x in tl("year_built")))
    occ = tl("occupancy")
    check("no duplicate OTHER: occupancy levels are unique", len(occ) == len(set(occ)) and occ.count("OTHER") == 1)
    check("trade code (stored as numbers) gets Float rules", set(rules[rules.Factor == "trade_code"]["Rules type"]) == {"Float"})
    check("trade code over 255 levels: smallest grouped to OTHER", "OTHER" in tl("trade_code") and len(tl("trade_code")) <= 255)
    fc = lambda f: " | ".join(checks[checks.Factor == f].Finding)                     # noqa: E731
    check("tiny Unknown level flagged (employees, 0.3% junk)", "Unknown holds only" in fc("employees"), fc("employees"))
    for f in ["years_trading", "turnover", "year_built"]:
        pts = [p for p, _ in (tidy[f].get("points") or [])]
        txt = fc(f)
        check(f"data check and kept spikes agree: {f}",
              all(_fmt(p, tidy[f].get("commas", True)) in txt for p in pts) and len(pts) > 0, txt)
    html = (out / "review_ALL_test.html").read_text(encoding="utf-8")
    check("report includes feature importance", "Feature importance (all perils)" in html)
    check("report includes correlation", "Correlation between factors" in html)
    check("infinite values logged", "infinite values" in html)
    return eng, tidy


def test_filter_split():
    print("\nFiltered to RE, bars split by property type")
    configure("re_split", FILTER={"product": ["RE"]}, SPLIT_BY="property_type")
    quiet(rr.main)
    out = OUT / "re_split"
    html = (out / "review_ALL_test_RE.html").read_text(encoding="utf-8") if (out / "review_ALL_test_RE.html").exists() else ""
    check("filtered outputs carry the filter tag", (out / "radar_spec_test_RE.xlsx").exists() and bool(html))
    check("report title shows the filter", "product = RE" in html)
    check("bars split by property type", "Bars split by" in html and '"name":"Shop"' in html)
    rm = pd.read_excel(out / "radar_spec_test_RE.xlsx", "Read me", keep_default_na=False)
    check("spec says it was built on filtered data", str(rm.iloc[0, 0]) == "!")
    print("\nFiltered to RPO (years trading has no values there; no AD claims, importance on)")
    configure("rpo", FILTER={"product": ["RPO"]}, OUTPUTS={**BASE["OUTPUTS"], "peril_reports": True},
              REPORT_SECTIONS={**BASE["REPORT_SECTIONS"], "importance": True, "raw_view": False, "tidy_view": False,
                               "group_view": False, "stability": False})
    try:
        quiet(rr.main)
        ran = True
    except Exception as e:
        ran, err = False, f"{type(e).__name__}: {e}"
    check("a peril with no claims doesn't stop the run (importance on)", ran, "" if ran else err)
    out = OUT / "rpo"
    html = (out / "review_ALL_test_RPO.html").read_text(encoding="utf-8") if ran else ""
    ad = (out / "review_AD_test_RPO.html").read_text(encoding="utf-8") if ran else ""
    check("the peril with no claims says importance wasn't estimated",
          "Not estimated" in ad and "AD (no claim cost in this data)" in html)
    check("a factor with no values left is skipped with a note",
          "years_trading: no values where it applies" in html and 'id="f-years-trading"' not in html)


# =============================================================================== baseline

def baseline_files(eng, tidy):
    """This run's results as CSV text: spec sheets, correlation, and every factor's groups for every peril."""
    out, files = OUT / "default", {}
    spec = pd.ExcelFile(out / "radar_spec_test.xlsx")
    for sh in ["Levels", "Banding rules", "Formulas", "Factor checks"]:
        files[f"spec_{sh.replace(' ', '_').lower()}.csv"] = pd.read_excel(spec, sh, keep_default_na=False).to_csv(
            index=False, float_format="%.6g")
    files["correlation.csv"] = pd.read_csv(out / "correlation_test.csv").to_csv(index=False, float_format="%.6g")
    cat = {f: f in BASE["CATEGORICAL"] for f in BASE["FACTORS"]}
    from review import raw_spec
    rows = []
    for peril in ["ALL"] + BASE["PERILS"]:
        for f in BASE["FACTORS"]:
            raw = raw_spec(eng, f, cat[f], tidy[f].get("reserved"), "exposure", rr.S)
            v = factor_views(eng, f, cat[f], tidy[f], raw, peril, rr.CFG, rr.S)
            t = v["tabs"]["group"][0]
            for lvl, r in t.iterrows():
                rows.append((peril, f, lvl, r.exposure, r.claims, r.bc_rel, v["base"]))
    files["groups.csv"] = pd.DataFrame(rows, columns=["peril", "factor", "level", "exposure", "claims", "bc_rel",
                                                      "base"]).to_csv(index=False, float_format="%.6g")
    return files


def compare_baseline(files, save):
    print("\nBaseline")
    if save:
        if BASELINE.exists():
            shutil.rmtree(BASELINE)
        BASELINE.mkdir(parents=True)
        for name, text in files.items():
            (BASELINE / name).write_text(text, encoding="utf-8")
        print(f"  saved {len(files)} files to {BASELINE}")
        return
    if not BASELINE.exists():
        print("  no baseline yet - run with --save-baseline once you're happy with the results")
        return
    for name, text in files.items():
        old = (BASELINE / name).read_text(encoding="utf-8") if (BASELINE / name).exists() else ""
        if old == text:
            check(f"unchanged from baseline: {name}", True)
        else:
            a, b = old.splitlines(), text.splitlines()
            diff = [f"line {i + 1}: was  {x}\n                 now  {y}" for i, (x, y) in enumerate(zip(a, b)) if x != y][:3]
            if len(a) != len(b):
                diff.append(f"{len(a)} lines before, {len(b)} now")
            check(f"unchanged from baseline: {name}", False, "\n        ".join(diff))


def main(save=False):
    if not DATA.exists():
        from make_test_data import make
        print(f"making test data: {make()}")
    OUT.mkdir(exist_ok=True)
    test_config()
    eng, tidy = test_default()
    files = baseline_files(eng, tidy)
    test_filter_split()
    compare_baseline(files, save)
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    for n in failed:
        print(f"  FAILED: {n}")
    return not failed


if __name__ == "__main__":
    ok = main(save="--save-baseline" in sys.argv)
    raise SystemExit(0 if ok else 1)
