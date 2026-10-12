"""
Feature importance and correlation between factors.

Importance  A gradient boosting model (LightGBM, Tweedie) is fitted to burning cost (claim cost / exposure,
            weighted by exposure) using every factor at once, plus product and the cover flags. Each
            factor's importance is how much worse the model predicts held-out data when that factor's
            values are shuffled (permutation importance), measured as deviance lost, as a % of the
            deviance of a prediction that is just the average. Unlike one-way relativity spreads, this
            allows for the other factors, so a factor that only looks strong because it moves with
            another one scores low.
            k-fold cross-validation: the data is split into `folds` parts; each part is held out once and
            predicted by a model trained on the rest (early stopping stops each model before it over-fits).
            The bars pool all folds; the whiskers show the range across folds.
            A column of random numbers (_noise_) is the benchmark: a factor scoring no better than noise
            has no detectable signal for this peril.
            Factors with an N/A level are shuffled only where they apply, so the score is what they add
            beyond the product / section split (which Emblem carries, with the N/A level fixed at 0).
            Without LightGBM installed, scikit-learn's HistGradientBoostingRegressor (Poisson) is used.

Correlation Exposure-weighted Cramér's V (bias-corrected) between every pair of factors, using the
            indicative groups for all perils combined for numeric factors and the suggested levels for
            categorical ones (N/A and Unknown as levels), plus product and the cover flags.
            0 = unrelated, 1 = one factor determines the other. High values mean the factors carry
            overlapping information: Emblem may alias them or give unstable parameters.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from banding import HtmlReport, _esc, _slug

C_TOP, C_LOW, C_CONTEXT, C_NOISE = "#2a78d6", "#c9c7bf", "#8fb3e0", "#e34948"


# =============================================================================== importance

def _tweedie_dev(y, mu, w, p):
    mu = np.clip(mu, 1e-9, None)
    y = np.clip(y, 0, None)
    a = np.where(y > 0, y ** (2 - p) / ((1 - p) * (2 - p)), 0.0)
    d = 2 * (a - y * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p))
    return float(np.sum(w * d) / np.sum(w))


def _design(df, features, cat_features):
    """Model matrix: numeric factors as values (missing stays missing), categoricals as pandas categories."""
    X = pd.DataFrame(index=df.index)
    for f in features:
        if f in cat_features:
            X[f] = df[f].astype("string").fillna("Unknown").astype("category")
        else:
            X[f] = pd.to_numeric(df[f], errors="coerce").astype(float)
    return X


def _min_leaf(n_rows, n_claims, st):
    """Fewest rows per leaf so that a leaf holds about min_claims_leaf claims on average."""
    if n_claims <= 0:
        return max(50, n_rows // 20)
    return int(np.clip(st["min_claims_leaf"] * n_rows / n_claims, 50, max(50, n_rows // 20)))


def _codes(X, cats):
    X = X.copy()
    for c in cats:                                    # ordinal codes; sklearn handles up to 255 categories
        X[c] = X[c].cat.codes.where(X[c].cat.codes < 254, 254).astype(float)
    return X


def _fit(X, y, w, cats, st, seed, min_leaf):
    """Gradient boosting with early stopping on 15% of the training rows (so it stops before over-fitting)."""
    val = np.random.default_rng(seed).random(len(X)) < 0.15
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=st["power"], n_estimators=st["max_trees"],
                              learning_rate=st["learning_rate"], num_leaves=st["leaves"], min_child_samples=min_leaf,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              max_cat_to_onehot=8, cat_smooth=50, cat_l2=10, min_data_per_group=min_leaf,
                              max_cat_threshold=16, random_state=seed, verbose=-1)
        with warnings.catch_warnings():               # newer LightGBM prefers eval_X / eval_y; both work
            warnings.simplefilter("ignore")
            m.fit(X[~val], y[~val], sample_weight=w[~val], eval_set=[(X[val], y[val])],
                  eval_sample_weight=[w[val]], callbacks=[lgb.early_stopping(st["early_stop"], verbose=False)])
        return m, "LightGBM (Tweedie, power %.1f)" % st["power"], int(m.best_iteration_ or st["max_trees"])
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(loss="poisson", max_iter=st["max_trees"], learning_rate=st["learning_rate"],
                                          max_leaf_nodes=st["leaves"], min_samples_leaf=min_leaf,
                                          categorical_features=[c in cats for c in X.columns],
                                          early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=st["early_stop"], random_state=seed)
        m.fit(_codes(X, cats), y, sample_weight=w)
        m._cats = list(cats)
        return m, "scikit-learn HistGradientBoosting (Poisson)", int(m.n_iter_)


def _predict(m, X):
    return m.predict(_codes(X, m._cats) if hasattr(m, "_cats") else X)


def gbm_importance(df, features, cat_features, claims, cost, expo, st, context=(), reserved=None):
    """
    Permutation importance for one peril, with k-fold cross-validation: every row is held out once.
    Importance = increase in held-out deviance when the factor is shuffled, as a % of the deviance of an
    average-only prediction (pooled over the folds; whiskers = range across folds).
    reserved: {factor: N/A code}. Those factors are shuffled only where they apply, so the score is what
    they add beyond the product / section split (which Emblem carries with the N/A level fixed at 0).
    Returns (table, info): table index = feature; importance, lo, hi, share (% of the top factor),
    below_noise, context.
    """
    reserved = reserved or {}
    seed = st["seed"]
    rng = np.random.default_rng(seed)
    d = df[df[expo] > 0]
    if st.get("max_rows") and len(d) > st["max_rows"]:
        d = d.sample(st["max_rows"], random_state=seed)
    feats = list(features) + [c for c in context if c not in features] + ["_noise_"]
    cats = [c for c in feats if c in cat_features or c in context]
    d = d.assign(_noise_=rng.random(len(d)))
    X = _design(d, feats, cats)
    y = (d[cost] / d[expo]).to_numpy(float)
    w = d[expo].to_numpy(float)
    n_claims = float(d[claims].sum())
    fold = rng.integers(0, st["folds"], len(d))
    applies = {}
    for f in feats:                                   # rows where the factor applies (not its N/A code)
        r = reserved.get(f)
        if r is None:
            applies[f] = np.ones(len(d), bool)
        elif f in cats:
            applies[f] = (X[f].astype(str) != str(r)).to_numpy()
        else:
            applies[f] = ~np.isclose(X[f].to_numpy(float), float(r), equal_nan=False)
    sums = {f: np.zeros(st["folds"]) for f in feats}  # weighted deviance totals per fold
    model_dev, null_dev, wsum, trees = np.zeros(st["folds"]), np.zeros(st["folds"]), np.zeros(st["folds"]), []
    method = ""
    for k in range(st["folds"]):
        te = fold == k
        tr = ~te
        m, method, nt = _fit(X[tr], y[tr], w[tr], cats, st, seed + k, _min_leaf(tr.sum(), d[claims][tr].sum(), st))
        trees.append(nt)
        Xte, yte, wte = X[te], y[te], w[te]
        wsum[k] = wte.sum()
        model_dev[k] = _tweedie_dev(yte, _predict(m, Xte), wte, st["power"]) * wsum[k]
        mean = np.sum(y[tr] * w[tr]) / np.sum(w[tr])
        null_dev[k] = _tweedie_dev(yte, np.full(len(yte), mean), wte, st["power"]) * wsum[k]
        for f in feats:
            a = applies[f][te]
            tot = 0.0
            for j in range(st["perms"]):
                Xp = Xte.copy()
                vals = Xp[f].to_numpy(copy=True)
                idx = np.flatnonzero(a)
                vals[idx] = vals[np.random.default_rng(seed + 100 * k + j).permutation(idx)]
                Xp[f] = pd.Categorical(vals, categories=Xte[f].cat.categories) if f in cats else vals
                tot += _tweedie_dev(yte, _predict(m, Xp), wte, st["power"]) * wsum[k]
            sums[f][k] = tot / st["perms"]
    rows = {}
    for f in feats:
        per_fold = 100 * (sums[f] - model_dev) / null_dev
        rows[f] = {"importance": 100 * (sums[f].sum() - model_dev.sum()) / null_dev.sum(),
                   "lo": per_fold.min(), "hi": per_fold.max()}
    t = pd.DataFrame(rows).T
    noise = max(float(t.loc["_noise_", "hi"]), 0.0)
    t["below_noise"] = t.importance <= noise
    t["context"] = [f in context for f in t.index]
    top = t.loc[~t.context & (t.index != "_noise_"), "importance"].max()
    t["share"] = 100 * t.importance.clip(lower=0) / top if top and top > 0 else 0.0
    t = t.sort_values("importance", ascending=False)
    explained = 100 * (1 - model_dev.sum() / null_dev.sum())
    warn = []
    if n_claims < st["min_claims_warn"]:
        warn.append(f"Only {n_claims:,.0f} claims: importances are rough — rely on the noise benchmark and the "
                    f"whiskers.")
    if explained <= 0:
        warn.append("The model as a whole predicts held-out data no better than the overall average, so treat any "
                    "factor above the noise line as weak evidence.")
    info = dict(method=method, rows=len(d), claims=n_claims, noise=noise, warn=" ".join(warn) or None,
                explained=explained, trees=int(np.median(trees)), folds=st["folds"])
    return t, info


def plot_importance(t, info, title):
    t = t.iloc[::-1]                                  # largest at the top
    names = [("(noise benchmark)" if i == "_noise_" else i) for i in t.index]
    col = [C_NOISE if i == "_noise_" else C_LOW if b else C_CONTEXT if c else C_TOP
           for i, b, c in zip(t.index, t.below_noise, t.context)]
    # axis fits the candidate factors; a context flag far beyond them (e.g. product for a peril only one
    # product covers) is cut off at the edge and labelled with its value
    cand = t[~t.context]
    hi = max(float(cand.hi.max()), float(cand.importance.max()), info["noise"], 1e-6) * 1.15
    lo = min(float(t.lo.min()), 0.0)
    lo = max(lo, -0.5 * hi)
    over = t.importance > hi
    fig = go.Figure(go.Bar(
        x=t.importance.clip(upper=hi), y=names, orientation="h", marker=dict(color=col),
        error_x=dict(type="data", symmetric=False, array=(t.hi.clip(upper=hi) - t.importance.clip(upper=hi)).clip(lower=0),
                     arrayminus=(t.importance.clip(upper=hi) - t.lo).clip(lower=0), color="#5f5e5a", thickness=1,
                     width=3),
        text=[f"{v:,.2f} →" if o else "" for v, o in zip(t.importance, over)], textposition="inside",
        insidetextanchor="end", textfont=dict(color="white", size=11),
        customdata=np.c_[t.lo, t.hi, t.importance],
        hovertemplate="%{y}: %{customdata[2]:.3f} (folds %{customdata[0]:.3f} to %{customdata[1]:.3f})"
                      "<extra></extra>"))
    fig.add_vline(x=info["noise"], line=dict(color=C_NOISE, dash="dash", width=1))
    fig.update_layout(title=dict(text=title, y=0.98, yanchor="top"), template="plotly_white",
                      height=26 * len(t) + 130, margin=dict(l=10, r=30, t=60, b=50), showlegend=False)
    fig.update_xaxes(title_text="Deviance lost when the factor is shuffled (% of an average-only model's deviance)",
                     zeroline=True, range=[lo, hi])
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    return fig


def importance_html(t, info, scope, fig_html):
    above = [i for i in t.index if not t.loc[i, "below_noise"] and not t.loc[i, "context"] and i != "_noise_"]
    below = [i for i in t.index if t.loc[i, "below_noise"] and not t.loc[i, "context"] and i != "_noise_"]
    html = (f"<h2>Feature importance ({_esc(scope)})</h2><p class='note'>A gradient boosting model of burning cost "
            f"using every factor at once ({_esc(info['method'])}; {info['rows']:,} rows, {info['claims']:,.0f} "
            f"claims, {info['folds']}-fold cross-validation, about {info['trees']} trees). It explains "
            f"{info['explained']:.1f}% of the held-out deviance of an average-only prediction. Importance = the "
            f"deviance it loses when a factor is shuffled, allowing for the other factors. Bars pool the folds; "
            f"whiskers show the range across folds. The dashed line is a column of random numbers — grey bars do "
            f"no better than it. Light blue = product / cover flags (context, not candidate factors). Factors "
            f"with an N/A level are shuffled only where they apply. For screening which factors to take into "
            f"Emblem, not a model.</p>")
    if info.get("warn"):
        html += f"<p class='note'><b>Note:</b> {_esc(info['warn'])}</p>"
    html += fig_html
    html += ("<p class='note'><b>Above noise:</b> " + (", ".join(_esc(x) for x in above) or "none")
             + "<br><b>No better than noise:</b> " + (", ".join(_esc(x) for x in below) or "none") + "</p>")
    return html


def importance_cross_table(results, factors):
    """Factor x peril: importance as % of the top factor in that peril; below-noise shown faded."""
    perils = list(results)
    head = "<th>Factor</th>" + "".join(f"<th>{_esc(p)}</th>" for p in perils)
    rows = []
    order = sorted(factors, key=lambda f: -np.nanmean([results[p][0].share.get(f, np.nan) for p in perils]))
    for f in order:
        cells = []
        for p in perils:
            t = results[p][0]
            if f not in t.index or not np.isfinite(t.loc[f, "share"]):
                cells.append("<td>–</td>")
                continue
            v, low = t.loc[f, "share"], bool(t.loc[f, "below_noise"])
            a = 0 if low else min(1, v / 100)
            r, g, b = [int(round(x + (z - x) * a)) for x, z in zip((247, 246, 243), (85, 146, 222))]
            style = f"background:rgb({r},{g},{b})" + (";color:#a3a29c" if low else "")
            cells.append(f'<td class="h" style="{style}">{v:.0f}{"" if not low else "·"}</td>')
        rows.append(f'<tr><td><a href="#{_slug(f)}">{_esc(f)}</a></td>{"".join(cells)}</tr>')
    return ("<h3>Importance by peril</h3><p class='note'>Each peril's own model: importance as a % of that "
            "peril's most important factor. Faded with a dot = no better than the noise benchmark for that "
            "peril.</p>"
            f'<div class="tbl heat"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody>'
            "</table></div>")


# =============================================================================== correlation

def _level_codes(s):
    s = pd.Series(s)
    if isinstance(s.dtype, pd.CategoricalDtype) and not s.isna().any():      # level labels: codes already there
        return s.cat.codes.to_numpy().astype(np.int64), len(s.cat.categories)
    codes, uniq = pd.factorize(s.astype(str), sort=False)
    return codes, len(uniq)


def _v_from_codes(ca, ka, cb, kb, w):
    """Bias-corrected Cramér's V (Bergsma 2013) on an exposure-weighted table scaled to the row count."""
    n = len(ca)
    ct = np.bincount(ca * kb + cb, weights=w, minlength=ka * kb).reshape(ka, kb)
    ct = ct[ct.sum(1) > 0][:, ct.sum(0) > 0]
    r, k = ct.shape
    if r < 2 or k < 2 or n < 3:
        return np.nan
    ct = ct / ct.sum() * n
    exp = np.outer(ct.sum(1), ct.sum(0)) / n
    phi2 = ((ct - exp) ** 2 / np.where(exp > 0, exp, 1)).sum() / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    den = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / den)) if den > 0 else np.nan


def cramers_v_weighted(a, b, w):
    """Cramér's V for one pair of label Series, weighted by exposure w."""
    (ca, ka), (cb, kb) = _level_codes(a), _level_codes(b)
    return _v_from_codes(ca, ka, cb, kb, np.asarray(w, float))


def correlation_matrix(labels, w):
    """labels: {name: Series of level labels}. Returns a symmetric DataFrame of Cramér's V.
    Each factor's levels are coded once and every pair's table is built with one bincount, so even
    60 factors (1,770 pairs) on a million rows take seconds rather than minutes."""
    names = list(labels)
    codes = {f: _level_codes(labels[f]) for f in names}
    w = np.asarray(w, float)
    m = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = _v_from_codes(*codes[a], *codes[b], w)
            m.loc[a, b] = m.loc[b, a] = v
    return m


def correlation_pairs(m, threshold, context=()):
    """Every pair with V at or above threshold, strongest first; context = product / cover flags."""
    rows = []
    names = list(m.index)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = m.loc[a, b]
            if np.isfinite(v) and v >= threshold:
                rows.append((a, b, float(v), a in context or b in context))
    return pd.DataFrame(rows, columns=["a", "b", "v", "context"]).sort_values("v", ascending=False,
                                                                             ignore_index=True)


def correlation_clusters(pairs):
    """Groups of factors linked, directly or through each other, by a listed pair (context pairs left out).
    Returns [(members, strongest V)] for groups of two or more, strongest first."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, v, ctx in pairs.itertuples(index=False):
        if not ctx:
            parent[find(a)] = find(b)
    groups = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    out = []
    for members in groups.values():
        if len(members) > 1:
            inside = pairs[pairs.a.isin(members) & pairs.b.isin(members) & ~pairs.context]
            out.append((members, float(inside.v.max())))
    return sorted(out, key=lambda g: -g[1])


def plot_correlation(m, title):
    n = len(m)
    z = m.to_numpy(float).copy()
    z[np.triu_indices(n, 0)] = np.nan                # lower triangle only
    txt = np.where(np.isnan(z), "", np.vectorize(lambda x: f"{x:.2f}")(np.nan_to_num(z)))
    fig = go.Figure(go.Heatmap(z=z, x=list(m.columns), y=list(m.index), zmin=0, zmax=1, text=txt,
                               texttemplate="%{text}", textfont=dict(size=10), hoverongaps=False,
                               colorscale=[[0, "#f7f6f3"], [0.3, "#b9d3f3"], [0.7, "#2a78d6"], [1, "#0c2e55"]],
                               colorbar=dict(title="V", thickness=12),
                               hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
    side = max(320, 34 * n)
    fig.update_layout(title=dict(text=title, y=0.98, yanchor="top"), template="plotly_white",
                      height=side + 200, margin=dict(l=10, r=10, t=60, b=10))
    fig.update_xaxes(side="bottom", tickangle=-40, automargin=True, showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(autorange="reversed", automargin=True, showgrid=False, tickfont=dict(size=11))
    return fig


def _imp_text(f, importance, context):
    if f in context:
        return "context"
    if not importance or f not in importance:
        return "–"
    share, low = importance[f]
    return f"{share:.0f}" + (" (noise)" if low else "")


def _name_link(f, context):
    return _esc(f) if f in context else f'<a href="#{_slug(f)}">{_esc(f)}</a>'


def correlation_html(m, context, flag=0.3, strong=0.7, importance=None, fig_html=None, csv_name=None):
    """
    Correlation section, built to stay readable with many factors:
      1. pairs of factors with V >= flag, strongest first, with each factor's importance beside it
      2. groups of factors linked through those pairs
      3. links with product / cover flags (collapsed - mostly expected N/A structure)
      4. the full matrix (collapsed), only when fig_html is given; every pair is also in the CSV
    importance: {factor: (% of top factor, below noise)} from the all-perils model, or None.
    """
    pairs = correlation_pairs(m, flag, context)
    main, ctx = pairs[~pairs.context], pairs[pairs.context]
    n_f = sum(1 for f in m.index if f not in context)
    html = ("<h2>Correlation between factors</h2><p class='note'>Cramér's V between every pair of factors "
            "(exposure-weighted, bias-corrected): 0 = unrelated, 1 = one determines the other. It's measured on "
            "the indicative groups for all perils (numeric factors) and the suggested levels (categorical factors), "
            "with N/A and Unknown as levels. Factors with a high V carry overlapping information: in Emblem they may "
            "alias or give unstable parameters, and one factor's one-way pattern can be the other's effect. "
            f"{n_f} factors, {n_f * (n_f - 1) // 2:,} pairs; only pairs with V of {flag:.2f} or more are listed "
            "(CORRELATION['flag'])" + (f"; every pair is in <code>{_esc(csv_name)}</code>" if csv_name else "")
            + ".</p>")
    if len(main):
        imp_note = (" Importance is from the all-perils model (% of the top factor; '(noise)' = no better than "
                    "random numbers): where two factors overlap, the more important one usually carries the "
                    "effect." if importance else "")
        rows = "".join(
            f"<tr><td>{_name_link(a, context)}</td><td>{_name_link(b, context)}</td><td>{v:.2f}</td>"
            f"<td>{'Strong' if v >= strong else 'Moderate'}</td>"
            + (f"<td>{_imp_text(a, importance, context)}</td><td>{_imp_text(b, importance, context)}</td>"
               if importance else "") + "</tr>"
            for a, b, v, _ in main.itertuples(index=False))
        head = ("<th>Factor</th><th>Factor</th><th>V</th><th>Strength</th>"
                + ("<th>Importance (1st)</th><th>Importance (2nd)</th>" if importance else ""))
        html += (f"<h3>Linked pairs ({len(main)})</h3><p class='note'>Strong (V of {strong:.1f} or more) = likely "
                 f"to alias or be unstable together in Emblem; moderate = worth checking together.{imp_note}</p>"
                 f"<div class='tbl'><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>")
        groups = correlation_clusters(pairs)
        if groups:
            items = []
            for i, (members, vmax) in enumerate(groups, start=1):
                if importance:
                    members = sorted(members, key=lambda f: -(importance.get(f, (0, True))[0] or 0))
                names = ", ".join(_name_link(f, context)
                                  + (f" ({_imp_text(f, importance, context)})" if importance else "")
                                  for f in members)
                items.append(f"<li><b>Group {i}</b> ({len(members)} factors, strongest link {vmax:.2f}): {names}</li>")
            html += ("<h3>Groups of linked factors</h3><p class='note'>Factors joined by the pairs above, directly or "
                     "through each other. Each group is a set to look at together: often one or two carry the "
                     "effect and the rest add little." + (" Listed most important first." if importance else "")
                     + "</p><ul class='checks'>" + "".join(items) + "</ul>")
    else:
        html += f"<p class='note'>No pair of factors has V of {flag:.2f} or more.</p>"
    if len(ctx):
        rows = "".join(f"<tr><td>{_name_link(a, context)}</td><td>{_name_link(b, context)}</td><td>{v:.2f}</td></tr>"
                       for a, b, v, _ in ctx.itertuples(index=False))
        html += HtmlReport._blk(
            f"Links with product and the cover flags ({len(ctx)})",
            "<p class='note'>A factor that only applies to some products or sections (an N/A level) always links "
            "with product or a cover flag; that's expected, and handled by fixing the N/A level in Emblem. A strong "
            "link without an N/A level means the factor mostly separates products.</p>"
            f"<div class='tbl'><table><thead><tr><th>Factor</th><th>Product / cover</th><th>V</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>", open_=False)
    if fig_html:
        html += HtmlReport._blk("Full matrix", fig_html, open_=False)
    return html
