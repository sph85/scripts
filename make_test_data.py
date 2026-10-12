"""
Synthetic SME property data for the regression tests: every awkward case the review has had to handle.
Fixed seed, so the data (and every result built on it) is the same each time.

    python tests/make_test_data.py        writes tests/data/test_data.xpt  (60,000 rows, SAS transport format)

What's in it (column names in capitals, as SAS stores them):
    product              RPO / CPO / RE
    Buildings_Cover, Contents_Cover   Y/N; sums insured are 0 / blank where the cover is N (structural)
    sum_insured_buildings, sum_insured_contents   thousands of distinct values (binned views)
    turnover             a default of 100,000 on 7% (a spike), 8% missing
    years_trading        blank for RPO (APPLIES_TO), 12% at 0 (a spike), 999 as junk
    employees            9999 as junk on 0.3% (a tiny Unknown level)
    trade_code           numeric codes, 380 of them (over Emblem's 255 limit; Float rules in Radar)
    construction         text with placeholders ("UNKNOWN", blank)
    occupancy            text that already contains a level called OTHER
    year_built           years (no thousands separators), 1900 as a default, a few 0 placeholders
    claims_per_property  a ratio of two counts: 70% at 0, decimals, and division by zero (infinite) - binned
    claims_per_location  a similar ratio with few distinct decimals (1/7, 1/3, 2/3...) - one level per value
    property_type        for SPLIT_BY, with some blanks
    exposure, policy_year, No_Of_Claims_<peril>, Claim_Cost_<peril> for six perils, with a few real effects
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

PATH = Path(__file__).resolve().parent / "data" / "test_data.xpt"


def make(path=PATH, n=60_000, seed=11):
    rng = np.random.default_rng(seed)
    prod = rng.choice(["RPO", "CPO", "RE"], n, p=[.45, .35, .20])

    def yn(p):
        return np.where(rng.random(n) < p, "Y", "N")

    bc = np.where(prod == "RE", yn(.6), yn(.95))
    cc = np.select([prod == "RPO", prod == "CPO"], [yn(.3), yn(.5)], yn(.95))
    sib = np.where(bc == "Y", np.round(np.exp(rng.normal(12.8, 0.8, n)), -3), 0.0)
    sic = np.where(cc == "Y", np.round(np.exp(rng.normal(10.5, 1.0, n)), -2), np.nan)
    yt = np.where(prod == "RPO", np.nan, rng.integers(1, 60, n).astype(float))
    yt[(prod != "RPO") & (rng.random(n) < .12)] = 0
    yt[(prod != "RPO") & (rng.random(n) < .01)] = 999
    to = np.round(np.exp(rng.normal(12, 1.1, n)), -2)
    to[rng.random(n) < .07] = 100000
    to[rng.random(n) < .08] = np.nan
    emp = rng.integers(0, 40, n).astype(float)
    emp[rng.random(n) < .003] = 9999
    tc = np.where(prod == "RPO", rng.integers(900, 960, n), rng.integers(100, 420, n)).astype(float)
    cons = rng.choice(["Standard", "Non-standard", "Timber", "UNKNOWN", ""], n, p=[.7, .15, .05, .06, .04])
    occ = np.select([prod == "RPO", prod == "CPO"],
                    [rng.choice(["Tenanted residential", "Vacant", "Holiday let"], n, p=[.8, .1, .1]),
                     rng.choice(["Office", "Retail", "Industrial", "Vacant", "OTHER"], n, p=[.3, .3, .2, .1, .1])],
                    rng.choice(["Retail", "Office", "Leisure"], n))
    yb = rng.integers(1850, 2026, n).astype(float)
    yb[rng.random(n) < .06] = 1900
    yb[rng.random(n) < .005] = 0
    props = np.where(rng.random(n) < .7, 1, rng.integers(2, 60, n)).astype(float)
    props[rng.random(n) < .003] = 0
    clm5 = rng.poisson(0.08 * np.maximum(props, 1)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cpp = clm5 / props                                        # inf where there are no properties
    locs = np.where(rng.random(n) < .7, 1, rng.integers(2, 8, n)).astype(float)
    cpl = rng.poisson(0.1 * locs).astype(float) / locs
    pt = rng.choice(["Shop", "Office", "Industrial unit", "Pub/restaurant", "Mixed use", "Flat above shop",
                     "Warehouse", "Surgery", "Garage", "Church hall", ""], n,
                    p=[.22, .2, .14, .1, .1, .07, .06, .04, .03, .02, .02])
    e = rng.uniform(.2, 1, n)
    yr = rng.choice([2021, 2022, 2023, 2024], n).astype(float)
    df = pd.DataFrame({"PRODUCT": prod, "BUILDINGS_COVER": bc, "CONTENTS_COVER": cc, "SUM_INSURED_BUILDINGS": sib,
                       "SUM_INSURED_CONTENTS": sic, "TURNOVER": to, "YEARS_TRADING": yt, "EMPLOYEES": emp,
                       "TRADE_CODE": tc, "CONSTRUCTION": cons, "OCCUPANCY": occ, "YEAR_BUILT": yb,
                       "CLAIMS_PER_PROPERTY": cpp, "CLAIMS_PER_LOCATION": cpl, "PROPERTY_TYPE": pt, "EXPOSURE": e, "POLICY_YEAR": yr})
    base = {"AD": .02, "Fire": .006, "EoW": .012, "Theft": .015, "Storm": .005, "Flood": .002}
    for p, f in base.items():
        lam = f * np.ones(n)
        if p == "AD":
            lam = np.where(prod == "RE", .04, 0) * np.where(yt == 0, 2.5, 1)
        if p == "Fire":
            lam = lam * np.where(cons == "Timber", 3, 1) * np.where(occ == "Industrial", 1.8, 1)
        if p == "EoW":
            lam = lam * np.clip(sib / 300000, 0.3, 3)
        if p == "Theft":
            lam = np.where(cc == "Y", .02, 0) * np.where(occ == "Retail", 2.5, 1)
        if p == "Storm":
            lam = lam * np.where(bc == "Y", 1.5, 0.4)
        k = rng.poisson(lam * e).astype(float)
        c = k * rng.gamma(2, 3000, n)
        k[k == 0] = np.nan                                        # blank = no claims, as in the extract
        c[np.isnan(k)] = np.nan
        df["NO_OF_CLAIMS_" + p.upper()] = k
        df["CLAIM_COST_" + p.upper()] = c
    path.parent.mkdir(parents=True, exist_ok=True)
    pyreadstat.write_xport(df.replace([np.inf, -np.inf], np.inf), str(path), file_format_version=8)
    return path


if __name__ == "__main__":
    print(f"written {make()}")
