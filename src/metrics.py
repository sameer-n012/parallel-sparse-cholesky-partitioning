from typing import Any, Optional

import numpy as np
import scipy.sparse as sp


def etree_from_L_csc(L_csc) -> list[int]:
    """
    Compute elimination tree parent[] from the sparsity of a Cholesky factor L.

    For a CSC L, parent[j] = min{i > j | L[i, j] != 0}, else -1 (root).
    This is a standard way to recover the etree from L's pattern.
    """

    n = L_csc.shape[0]
    parent = np.full(n, -1, dtype=int)
    indptr = L_csc.indptr
    indices = L_csc.indices

    for j in range(n):
        rows = indices[indptr[j] : indptr[j + 1]]
        # Exclude diagonal and anything above it.
        cand = rows[rows > j]
        if cand.size:
            parent[j] = int(cand.min())
    return parent.tolist()


def etree_stats(parent: list[int]) -> dict:

    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for i, p in enumerate(parent):
        if p is None or p < 0:
            roots.append(i)
        else:
            children[p].append(i)

    depth = np.full(n, -1, dtype=int)
    stack: list[int] = []
    for r in roots:
        depth[r] = 0
        stack.append(r)
        while stack:
            u = stack.pop()
            du = int(depth[u])
            for v in children[u]:
                depth[v] = du + 1
                stack.append(v)

    max_depth = int(depth.max(initial=0))
    widths = [0] * (max_depth + 1)
    for d in depth.tolist():
        if d >= 0:
            widths[d] += 1

    # widths[d] is number of nodes at depth d
    return {"n": n, "height": max_depth, "widths": widths}


def factor_metrics(*, A_csc, L_csc) -> dict[str, Any]:
    """
    Compute metrics from A and its Cholesky factor L.
    """

    n = A_csc.shape[0]
    # nnz in lower triangle of A including diagonal
    A_lower = sp.tril(A_csc, format="csc")
    nnz_A_lower = int(A_lower.nnz)

    nnz_L = int(L_csc.nnz)
    fill_in = nnz_L - nnz_A_lower

    col_nnz = np.diff(L_csc.indptr)
    max_col_nnz = int(col_nnz.max(initial=0))

    parent = etree_from_L_csc(L_csc)
    es = etree_stats(parent)

    return {
        "n": int(n),
        "nnz_A": int(A_csc.nnz),
        "nnz_A_lower": nnz_A_lower,
        "nnz_L": nnz_L,
        "fill_in": int(fill_in),
        "max_col_nnz_L": max_col_nnz,
        "etree_height": int(es["height"]),
        "etree_widths": es["widths"],
    }
