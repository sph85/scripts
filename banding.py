"""
Pre-Emblem variable review and banding.

Main entry points
-----------------
load_data(path, columns)          read .sas7bdat / .xpt / .csv / .parquet (column names case-insensitive)
prepare(df, ...)                  blank claims -> 0, drop zero exposure, floor negative cost
raw_view_table / plot_raw         every value (or ~100 unrounded bins) with a smoothed trend
peril_overview(...)               per factor: burning cost relativity by peril + peril mix
review_all(df, factors, ...)      loop over every factor: profile, charts, suggested bands
                                  (show=True: inline in Jupyter; report=HtmlReport(): HTML file)
review_variable(df, col, ...)     the same for one factor (use with spec=... to check an override)
apply_spec / validate / save_spec / label_table
                                  build the Emblem file, check it, and export the band definitions

Usually run via run_banding.py.

How bands are suggested
-----------------------
Numeric:
    1. Candidate cut points = ~20 equal-EXPOSURE quantiles, each snapped to a round number
       (leading digits 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5 ... 7.5, 8, 9), so 41,372 -> 40,000
       and 190,000 -> 200,000.
       Low-cardinality whole-number factors (<= 15 values) start with one band per value.
       A whole-number factor with >= 5% of exposure at its minimum (e.g. 0 years trading)
       keeps that value as its own band.
    2. The thinnest band (fewest claims) is merged into its thinner neighbour, repeatedly,
       until every band has >= min_claims and there are <= max_bands.
Categorical:
    Levels with >= min_claims claims and >= min_expo_share of exposure are kept;
    the rest are grouped into OTHER. Levels are ordered by exposure (code 1 = largest).

Conventions
-----------
- Numeric bands are lower-inclusive [a, b): edges [25, 30] give "<25", "25-29", "30+".
- Code 0 is always Missing; codes 1..k follow natural order.
- Relativities are relative to the whole-dataset average (1.00 = average).
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from IPython.display import Markdown, display
except ImportError:  # plain Python: fall back to printing
    Markdown = str
    display = print


# Chart colours: neutral exposure bars; fixed hue per measure (never reassigned)
C_EXPO, C_BC, C_FREQ, C_SEV, C_REF = "#c9c7bf", "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
BLUES = ["#b9d3f3", "#86b3ea", "#5592de", "#2a78d6", "#1d5fb0", "#14467f", "#0c2e55"]


# ---------------------------------------------------------------- inspection


# ------------------------------------------------------- candidate cut points

def _weighted_quantiles(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cw, x)


def quantile_edges(df, col, n_bands, expo="exposure"):
    """Equal-EXPOSURE bands (not equal row count)."""
    m = df[col].notna()
    qs = np.linspace(0, 1, n_bands + 1)[1:-1]
    return np.unique(_weighted_quantiles(df.loc[m, col].to_numpy(float), df.loc[m, expo].to_numpy(float), qs))


def _between(a, b):
    """
    The shortest round number r with a < r <= b: a cut point between two neighbouring decimal values.
    E.g. between 0 and 0.01695 -> 0.01; between 0.65 and 0.6667 -> 0.66. A short number survives being
    copied into Excel and Radar exactly, whereas 0.6666666666666666 can round up past the value it should
    sit below and put that value in the wrong band.
    """
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return b
    if b == 0:
        return 0.0
    for sf in range(1, 16):
        mag = 10 ** (math.floor(math.log10(abs(b))) - (sf - 1))
        r = float(f"{math.floor(b / mag) * mag:.{sf + 2}g}")
        if a < r <= b:
            return r
    return b


def find_points(v, w, point_min=0.02, point_ratio=3.0):
    """
    Spikes: values holding at least point_min of the exposure AND at least point_ratio times the average
    share of up to two neighbouring values on each side (so an even spread, like 60 years each with 1.7%,
    has no spikes, but 8% at 0 years trading does). Returns [(value, upper edge)], where the band
    [value, upper) holds only that value.
    """
    g = w.groupby(v).sum().sort_index()
    if len(g) < 2:
        return []
    share = (g / g.sum()).to_numpy()
    vals = g.index.to_numpy(float)
    whole = np.allclose(vals, np.round(vals))
    out = []
    for i, (x, sh) in enumerate(zip(vals, share)):
        if sh < point_min:
            continue
        nb = np.r_[share[max(0, i - 2):i], share[i + 1:i + 3]]
        if len(nb) and sh >= point_ratio * nb.mean():
            upper = x + 1 if whole else (_between(x, vals[i + 1]) if i + 1 < len(vals) else None)
            out.append((_tidy(x), None if upper is None else _tidy(upper)))
    return out


def _is_whole(s):
    v = s.dropna().to_numpy(float)
    return len(v) > 0 and np.allclose(v, np.round(v))


# ------------------------------------------------------------- band spec

def _fmt(x, commas=True):
    """1000000 -> '1,000,000'; 0.25 -> '0.25' (avoids scientific notation).
    commas=False for factors such as years: 2026 -> '2026', not '2,026'."""
    x = float(x)
    sep = "," if commas else ""
    return f"{x:{sep}.0f}" if x.is_integer() else f"{x:{sep}.4f}".rstrip("0").rstrip(".")


YEAR_RANGE = (1800, 2100)


def year_like(values, weights=None, share=0.99):
    """True for a factor holding years: whole numbers, with at least 99% of them (by exposure, when
    weights are given) between 1800 and 2100. A few placeholders such as 0 or 9999 don't stop it."""
    v = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    w = (pd.Series(1.0, index=v.index) if weights is None
         else pd.Series(np.asarray(weights, dtype=float)))
    ok = v.notna() & np.isfinite(v)
    if not ok.any():
        return False
    v, w = v[ok].astype(float), w[ok]
    if not np.allclose(v, np.round(v)):
        return False
    return bool(w[v.between(*YEAR_RANGE)].sum() >= share * w.sum())


def numeric_labels(edges, right=False, integer=True, data_min=None, commas=True):
    """Human-readable labels matching pd.cut intervals.
    data_min (whole-number factors): label the lowest band from the observed minimum,
    e.g. "0" or "1-3" instead of "<1" / "<4". The band itself still catches anything lower.
    commas=False: no thousands separators (years)."""
    def _fmt(x):
        return _fmt_num(x, commas)
    e = list(edges)
    if not e:
        return ["All"]
    if integer and not right and data_min is not None and data_min < e[0]:
        lo, hi = float(data_min), e[0] - 1
        first = _fmt(lo) if hi == lo else f"{_fmt(lo)}-{_fmt(hi)}"
        mid = [_fmt(a) if b - 1 == a else f"{_fmt(a)}-{_fmt(b - 1)}" for a, b in zip(e, e[1:])]
        return [first] + mid + [f"{_fmt(e[-1])}+"]
    if right:      # (a, b]
        mid = [f"{_fmt(a)}<x<={_fmt(b)}" for a, b in zip(e, e[1:])]
        return [f"<={_fmt(e[0])}"] + mid + [f">{_fmt(e[-1])}"]
    if integer:    # [a, b) on whole numbers -> a to b-1
        mid = [_fmt(a) if b - 1 == a else f"{_fmt(a)}-{_fmt(b - 1)}" for a, b in zip(e, e[1:])]
        return [f"<{_fmt(e[0])}"] + mid + [f"{_fmt(e[-1])}+"]
    mid = [f"{_fmt(a)}<=x<{_fmt(b)}" for a, b in zip(e, e[1:])]
    return [f"<{_fmt(e[0])}"] + mid + [f">={_fmt(e[-1])}"]


# ------------------------------------------------------- automated review


def plot_stability(rel, title, base=None):
    """One line per period (light = oldest, dark = newest)."""
    periods = list(rel.columns)
    idx = np.linspace(0, len(BLUES) - 1, len(periods)).round().astype(int) if len(periods) > 1 else [3]
    fig = go.Figure()
    for p, k in zip(periods, idx):
        name = str(_tidy(p)) if isinstance(p, (int, float, np.number)) else str(p)   # 2021.0 -> 2021
        fig.add_trace(go.Scatter(x=[str(i) for i in rel.index], y=rel[p], name=name, mode="lines+markers",
                                 line=dict(color=BLUES[k], width=2), marker=dict(size=7),
                                 hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=C_REF, width=1, dash="dot"))
    fig.update_layout(title=dict(text=title, y=0.97), template="plotly_white", hovermode="x unified",
                      height=400, yaxis_title="Relativity (1 = base)" if base else "Relativity (1 = average)",
                      yaxis_rangemode="tozero",
                      margin=dict(t=90, b=20, l=70, r=70),
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
    fig.update_xaxes(type="category", tickangle=-40 if len(rel) > 8 else 0, automargin=True)
    return fig


# ------------------------------------------------------- raw (granular) view


def _fmt_any(v, commas=True):
    try:
        return _fmt(v, commas)
    except (TypeError, ValueError):
        return str(v)


_fmt_num = _fmt


# ------------------------------------------------------- peril comparison


def _tidy(x):
    x = float(x)
    return int(x) if x.is_integer() else x


# ---------------------------------------------------------------- data loading

def load_data(path, columns=None, encoding=None):
    """
    Read a SAS dataset (.sas7bdat / .xpt), .csv or .parquet. Column names are matched
    case-insensitively (SAS names are often upper case) and returned exactly as requested.
    Uses pyreadstat for SAS files if installed (faster, reads only the columns needed).
    """
    path = str(path)
    ext = path.lower().rsplit(".", 1)[-1]
    want = list(dict.fromkeys(columns)) if columns else None

    def _match(available):
        lookup = {c.lower(): c for c in available}
        missing = [c for c in want if c.lower() not in lookup]
        if missing:
            raise KeyError(f"not found in {path}: {missing}\navailable: {sorted(available)}")
        return {lookup[c.lower()]: c for c in want}       # actual name -> requested name

    if ext in ("sas7bdat", "xpt"):
        try:
            import pyreadstat
            reader = pyreadstat.read_sas7bdat if ext == "sas7bdat" else pyreadstat.read_xport
            kw = {"encoding": encoding} if encoding else {}
            rename = None
            if want:
                _, meta = reader(path, metadataonly=True, **kw)
                rename = _match(meta.column_names)
                kw["usecols"] = list(rename)
            df, _ = reader(path, **kw)
        except ImportError:
            df = pd.read_sas(path, format="sas7bdat" if ext == "sas7bdat" else "xport",
                             encoding=encoding or "latin-1")
            rename = _match(df.columns) if want else None
    elif ext == "csv":
        df = pd.read_csv(path, low_memory=False)
        rename = _match(df.columns) if want else None
    elif ext == "parquet":
        df = pd.read_parquet(path)
        rename = _match(df.columns) if want else None
    else:
        raise ValueError(f"unsupported file type: {path}")
    if rename:
        df = df[list(rename)].rename(columns=rename)
    return df


def prepare(df, expo, claims, cost=None):
    """
    Basic cleaning for burning cost / Tweedie data. claims and cost can be a column name or a
    list (one per peril). Blank claims -> 0, zero/blank exposure dropped, negative cost -> 0.
    Returns (df, list_of_notes).
    """
    counts = [claims] if isinstance(claims, str) else list(claims)
    costs = [] if cost is None else ([cost] if isinstance(cost, str) else list(cost))
    notes, blanks = [], {}
    for c in counts + costs:
        n = int(df[c].isna().sum())
        if n:
            df[c] = df[c].fillna(0)
            blanks[c] = n
    if len(blanks) > 2:
        notes.append(f"Blank claim counts/costs set to 0 (no claims) in {len(blanks)} columns, "
                     f"{min(blanks.values()):,}–{max(blanks.values()):,} rows each")
    else:
        notes += [f"{n:,} rows with blank {c} set to 0 (no claims)" for c, n in blanks.items()]
    bad = df[expo].isna() | (df[expo] <= 0)
    if bad.any():
        notes.append(f"{int(bad.sum()):,} rows with zero, negative or blank {expo} removed")
        df = df[~bad].copy()
    for c in costs:
        neg = df[c] < 0
        if neg.any():
            notes.append(f"{int(neg.sum()):,} rows with negative {c} floored at 0 (total {df.loc[neg, c].sum():,.0f})")
            df.loc[neg, c] = 0
    for cc, kc in zip(counts, costs):
        odd = (df[cc] == 0) & (df[kc] > 0)
        if odd.any():
            notes.append(f"{int(odd.sum()):,} rows with {kc} > 0 but {cc} = 0 — check (e.g. reopened/IBNR claims)")
    if not notes:
        notes.append("no issues found")
    for n in notes:
        print("data check:", n)
    return df, notes


# ---------------------------------------------------------------- HTML report

_CSS = """
:root { --ink:#1f1f1d; --ink2:#55534e; --muted:#8a8984; --line:#e4e2dc; --bg:#ffffff; --panel:#f7f6f3;
        --accent:#2a78d6; --thin:#fdf3dc; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width:1150px; margin:0 auto; padding:24px 20px 80px; }
h1 { font-size:24px; margin:0 0 4px; }
h2 { font-size:20px; margin:0 0 10px; }
h3 { font-size:15px; margin:22px 0 8px; color:var(--ink2); }
.sub { color:var(--ink2); margin:0 0 20px; }
section { border-top:1px solid var(--line); padding-top:24px; margin-top:32px; }
.meta { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:8px 24px;
        background:var(--panel); padding:14px 16px; border-radius:6px; }
.meta div span { display:block; color:var(--muted); font-size:12px; }
.stats { display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 14px; }
.stat { background:var(--panel); border-radius:6px; padding:8px 12px; min-width:110px; }
.stat span { display:block; color:var(--muted); font-size:12px; }
.stat b { font-size:16px; font-weight:600; }
.tag { display:inline-block; font-size:12px; padding:1px 8px; border-radius:10px; background:var(--panel);
       color:var(--ink2); margin-left:8px; vertical-align:middle; font-weight:normal; }
.tbl { overflow-x:auto; }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; margin:4px 0 8px; }
th, td { padding:5px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th { color:var(--ink2); font-weight:600; font-size:12px; border-bottom:1px solid var(--muted); }
th:first-child, td:first-child { text-align:left; }
tr.thin td { background:var(--thin); }
code, pre { font-family:Consolas, "SFMono-Regular", Menlo, monospace; font-size:13px; }
pre { background:var(--panel); padding:10px 12px; border-radius:6px; white-space:pre-wrap; word-break:break-all; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
.top { font-size:12px; float:right; }
.note { color:var(--ink2); font-size:13px; }
ul.checks { margin:6px 0; padding-left:20px; }
details.blk { margin:14px 0; border:1px solid var(--line); border-radius:6px; padding:0 12px; }
details.blk > summary { cursor:pointer; padding:9px 0; font-weight:600; color:var(--ink2); }
details.blk[open] > summary { border-bottom:1px solid var(--line); margin-bottom:8px; }
.heat td.h { color:var(--ink); }
.cols2 { display:flex; flex-wrap:wrap; gap:24px; }
"""

_JS = """
document.addEventListener('toggle', function (e) {
  if (e.target.tagName === 'DETAILS' && e.target.open && window.Plotly) {
    e.target.querySelectorAll('.js-plotly-plot').forEach(function (p) { Plotly.Plots.resize(p); });
  }
}, true);
"""


def _esc(x):
    import html
    return html.escape(str(x))


def _cell(v, f=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if f:
        return format(v, f)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}".rstrip("0").rstrip(".") if v % 1 else f"{v:,.0f}"
    return _esc(v)


def _table(df, fmt=None, headers=None, row_class=None, index_name=None, links=None):
    fmt, headers = fmt or {}, headers or {}
    h = [f"<th>{_esc(index_name or df.index.name or '')}</th>"]
    h += [f"<th>{_esc(headers.get(c, c))}</th>" for c in df.columns]
    rows = []
    for i, (idx, r) in enumerate(df.iterrows()):
        cls = f' class="{row_class[i]}"' if row_class is not None and row_class[i] else ""
        first = f'<a href="#{links[idx]}">{_esc(idx)}</a>' if links else _esc(idx)
        cells = [f"<td>{first}</td>"] + [f"<td>{_cell(r[c], fmt.get(c))}</td>" for c in df.columns]
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f'<div class="tbl"><table><thead><tr>{"".join(h)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def heat_table(df, index_name="", fmt=".2f", links=None):
    """Table with cells shaded light-to-dark blue by value (log scale; NaN blank)."""
    v = np.log(df.astype(float).clip(lower=1))
    hi = np.nanmax(v.to_numpy()) if np.isfinite(v.to_numpy()).any() else 1
    def shade(x):
        if not np.isfinite(x):
            return ""
        t = 0 if hi <= 0 else min(1, x / hi)
        r, g, b = [int(round(a + (z - a) * t)) for a, z in zip((247, 246, 243), (85, 146, 222))]
        return f' style="background:rgb({r},{g},{b})"'
    h = f"<th>{_esc(index_name)}</th>" + "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    rows = []
    for idx, r in df.iterrows():
        first = f'<a href="#{links[idx]}">{_esc(idx)}</a>' if links else _esc(idx)
        cells = "".join(f'<td class="h"{shade(v.loc[idx, c])}>{_cell(r[c], fmt)}</td>' for c in df.columns)
        rows.append(f"<tr><td>{first}</td>{cells}</tr>")
    return f'<div class="tbl heat"><table><thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _slug(s):
    import re
    return "f-" + re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


class HtmlReport:
    """Collects review output and writes one self-contained HTML file (works offline)."""

    def __init__(self, title="Banding review", subtitle=""):
        self.title, self.subtitle = title, subtitle
        self.sections, self.summary_html = [], None
        self.meta, self.checks = {}, []

    @staticmethod
    def _blk(title, body, open_=True):
        return f'<details class="blk"{" open" if open_ else ""}><summary>{_esc(title)}</summary>{body}</details>'

    def add_block_section(self, col, blocks, intro=""):
        """Generic factor section: blocks = [(title, fig_or_html, note), ...]."""
        parts = []
        for title, obj, note in blocks:
            body = (f'<p class="note">{_esc(note)}</p>' if note else "") + (
                obj if isinstance(obj, str) else self._fig(obj))
            parts.append(self._blk(title, body))
        self.sections.append(f"""
<section id="{_slug(col)}">
  <a class="top" href="#top">back to top</a>
  <h2>{_esc(col)}</h2>{intro}
  {''.join(parts)}
</section>""")

    @staticmethod
    def _fig(fig):
        if fig is None:
            return ""
        return fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%",
                           config={"displaylogo": False, "responsive": True})


    def save(self, path, plotlyjs="inline"):
        """plotlyjs='inline' embeds plotly.js (~4.5MB, works offline); 'cdn' keeps the file small."""
        import datetime
        from plotly.offline import get_plotlyjs, get_plotlyjs_version
        js = (f"<script>{get_plotlyjs()}</script>" if plotlyjs == "inline"
              else f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>')
        meta = {"Generated": datetime.datetime.now().strftime("%d %b %Y %H:%M"), **self.meta}
        meta_html = "".join(f"<div><span>{_esc(k)}</span>{_esc(v)}</div>" for k, v in meta.items())
        checks = "".join(f"<li>{_esc(c)}</li>" for c in self.checks)
        summ = self.summary_html or ""
        html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(self.title)}</title><style>{_CSS}</style>{js}</head>
<body><main id="top">
<h1>{_esc(self.title)}</h1><p class="sub">{_esc(self.subtitle)}</p>
<div class="meta">{meta_html}</div>
{('<h3>Data checks</h3><ul class="checks">' + checks + '</ul>') if checks else ''}
<section style="border:0;margin-top:8px">{summ}</section>
{''.join(self.sections)}
</main><script>{_JS}</script></body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        return path


# ---------------------------------------------------------------- checks

