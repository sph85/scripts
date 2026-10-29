import numpy as np
import pandas as pd

def subgroup_search(d, target_col, depth=3, beam_width=50, top_k=25,
                    min_size=50, a=1.0):
    """Beam-search subgroup discovery for a 0/1 target.
    a=1.0 -> WRAcc (favours bigger groups); a=0.5 -> leans towards higher lift."""
    y = d[target_col].to_numpy()
    N, p0 = len(y), y.mean()

    # One selector per (factor, level), skipping levels below min_size
    selectors = []
    for c in d.columns.drop(target_col):
        for lvl in d[c].unique():
            m = (d[c] == lvl).to_numpy()
            if m.sum() >= min_size:
                selectors.append(((c, lvl), m))

    def quality(m):
        n = m.sum()
        return -np.inf if n < min_size else (n / N) ** a * (y[m].mean() - p0)

    results = {}
    beam = []
    for sel, m in selectors:
        r = ((sel,), m, quality(m))
        results[frozenset(r[0])] = r
        beam.append(r)

    for _ in range(depth - 1):
        beam = sorted(beam, key=lambda r: r[2], reverse=True)[:beam_width]
        candidates = []
        for rule, m, _ in beam:
            used = {c for c, _ in rule}
            for sel, sm in selectors:
                if sel[0] in used:
                    continue
                key = frozenset(rule + (sel,))
                if key in results:
                    continue
                nm = m & sm
                q = quality(nm)
                results[key] = (tuple(sorted(rule + (sel,), key=str)), nm, q)
                if q > -np.inf:
                    candidates.append(results[key])
        if not candidates:
            break
        beam = candidates

    top = sorted(results.values(), key=lambda r: r[2], reverse=True)[:top_k]
    rows = []
    for rule, m, q in top:
        n, pos = m.sum(), y[m].sum()
        rows.append({
            "rule": " AND ".join(f"{c} = {l}" for c, l in rule),
            "conditions": rule,
            "size": n,
            "positives": pos,
            "rate": pos / n,
            "lift": (pos / n) / p0,
            "share_of_positives": pos / y.sum(),
            "quality": q,
        })
    return pd.DataFrame(rows)


def rule_mask(d, conditions):
    m = np.ones(len(d), dtype=bool)
    for c, lvl in conditions:
        m &= (d[c] == lvl).to_numpy()
    return m