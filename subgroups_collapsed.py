import numpy as np
import pandas as pd
import scipy.sparse as sp

def subgroup_search_fast(d, target_col, depth=3, beam_width=30, top_k=25,
                         min_size=50, a=1.0):
    factors = [c for c in d.columns if c != target_col]

    # 1. Collapse identical rows to counts
    agg = (d.groupby(factors, dropna=False, observed=True, sort=False)[target_col]
             .agg(n="size", pos="sum").reset_index())
    w = agg["n"].to_numpy(dtype=float)
    p = agg["pos"].to_numpy(dtype=float)
    N, P = w.sum(), p.sum()
    p0 = P / N
    R = len(agg)

    # 2. Sparse one-hot: rows = unique combos, cols = (factor, level)
    blocks, labels, col_factor = [], [], []
    for fi, c in enumerate(factors):
        codes, uniques = pd.factorize(agg[c])
        blocks.append(sp.csr_matrix((np.ones(R), (np.arange(R), codes)),
                                    shape=(R, len(uniques))))
        labels += [(c, u) for u in uniques]
        col_factor += [fi] * len(uniques)
    X = sp.hstack(blocks, format="csr")
    Xc = X.tocsc()
    col_factor = np.array(col_factor)

    def qual(n, pos):
        with np.errstate(divide="ignore", invalid="ignore"):
            q = (n / N) ** a * (pos / n - p0)
        q[n < min_size] = -np.inf
        return q

    def col_rows(j):
        return Xc.indices[Xc.indptr[j]:Xc.indptr[j + 1]]

    # Depth 1: all single conditions at once
    n1, pos1 = X.T @ w, X.T @ p
    q1 = qual(n1, pos1)
    results = {frozenset([j]): (q1[j], n1[j], pos1[j])
               for j in np.flatnonzero(q1 > -np.inf)}

    beam = []
    for j in np.argsort(-q1)[:beam_width]:
        if q1[j] == -np.inf:
            break
        m = np.zeros(R, bool)
        m[col_rows(j)] = True
        beam.append(((j,), m))

    # Deeper levels: score every extension of each beam rule in one product
    for _ in range(depth - 1):
        cands = []
        for rule, m in beam:
            idx = np.flatnonzero(m)
            Xm = X[idx]
            n_ext = Xm.T @ w[idx]
            pos_ext = Xm.T @ p[idx]
            q = qual(n_ext, pos_ext)
            q[np.isin(col_factor, col_factor[list(rule)])] = -np.inf
            for j in np.flatnonzero(q > -np.inf):
                key = frozenset(rule + (j,))
                if key not in results:
                    results[key] = (q[j], n_ext[j], pos_ext[j])
                    cands.append((q[j], rule + (j,), m))
        if not cands:
            break
        cands.sort(key=lambda t: t[0], reverse=True)
        beam = []
        for _, rule, parent in cands[:beam_width]:
            rows = col_rows(rule[-1])
            nm = np.zeros(R, bool)
            nm[rows] = parent[rows]
            beam.append((rule, nm))

    top = sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
    out = []
    for key, (q, n, pos) in top:
        conds = tuple(sorted((labels[j] for j in key), key=str))
        out.append({
            "rule": " AND ".join(f"{c} = {l}" for c, l in conds),
            "conditions": conds,
            "size": int(n),
            "positives": int(pos),
            "rate": pos / n,
            "lift": (pos / n) / p0,
            "share_of_positives": pos / P,
            "quality": q,
        })
    return pd.DataFrame(out)