from pathlib import Path
from turtle import down
from typing import Optional

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import ssgetpy
import os

# MatrixKind = Literal[
#     "spd_suitesparse",  # Prefer matrices that are likely SPD (best-effort filter).
#     "symmetric_suitesparse",  # Symmetric (best-effort), may require diagonal shift to be SPD.
#     "laplacian_suitesparse",  # Build a (shifted) graph Laplacian from a SuiteSparse sparsity pattern.
#     "curated_spd",  # Small curated list of known-useful SPD-ish problems.
# ]


def _ssget_search(
    limit: int,
    dtype: str = "real",
    rows: Optional[tuple] = None,
    cols: Optional[tuple] = None,
    nnzs: Optional[tuple] = None,
    spd: Optional[bool] = None,
    d23: Optional[bool] = None,
    group: Optional[str] = None,
    kind: Optional[str] = None,
):
    # https://github.com/drdarshan/ssgetpy/blob/master/demo.ipynb
    if not rows:
        rows = (None, None)
    if not cols:
        cols = (None, None)
    if not nnzs:
        nnzs = (None, None)

    return ssgetpy.search(
        limit=limit,
        dtype=dtype,
        rowbounds=rows,
        colbounds=cols,
        nzbounds=nnzs,
        isspd=spd,
        is2d3d=d23,
        group=group,
        kind=kind
    )


# def _candidate_text(m) -> str:
#     # Best-effort extraction for filtering (works even if ssgetpy changes internals).
#     parts: list[str] = []
#     for attr in ("name", "group", "kind", "notes"):
#         v = getattr(m, attr, None)
#         if v:
#             parts.append(str(v))
#     parts.append(str(m))
#     return " ".join(parts).lower()


# def list_matrix_refs(
#     *,
#     kind: MatrixKind,
#     limit: int = 20,
#     search_limit: int = 2000,
# ) -> list[MatrixRef]:
#     """Return up to `limit` SuiteSparse problem names for a given matrix kind.

#     This is intentionally heuristic: it tries to find matrices that are relevant
#     to sparse SPD Cholesky behavior (PDE/structural/Laplacian-like).
#     """

#     if kind == "curated_spd":
#         # Keep this list small and easy to override; names are SuiteSparse problem names.
#         curated = [
#             "bcsstk18",
#             "bcsstk24",
#             "bcsstk27",
#             "bcsstk36",
#             "nd3k",
#             "nd6k",
#             "s1rmt3m1",
#             "s3rmt3m3",
#         ]
#         return [MatrixRef(name=n) for n in curated[:limit]]

#     mats = _ssget_search(limit=search_limit)

#     # Broad buckets by keyword; we keep them intentionally inclusive and then rely
#     # on downstream SPD checks/diagonal-shift.
#     if kind in ("spd_suitesparse", "symmetric_suitesparse"):
#         allow = [
#             "spd",
#             "posdef",
#             "positive definite",
#             "symmetric",
#             "struct",
#             "fem",
#             "pde",
#             "laplacian",
#             "stiffness",
#             "thermal",
#             "elastic",
#             "mechanics",
#             "optimization",
#             "gmr",
#             "markov",
#         ]
#     elif kind == "laplacian_suitesparse":
#         allow = [
#             "laplacian",
#             "graph",
#             "network",
#             "road",
#             "web",
#             "social",
#         ]
#     else:
#         raise ValueError(f"Unknown kind: {kind}")

#     refs: list[MatrixRef] = []
#     seen: set[str] = set()
#     for m in mats:
#         name = getattr(m, "name", None)
#         if not name or name in seen:
#             continue
#         txt = _candidate_text(m)
#         if any(k in txt for k in allow):
#             refs.append(MatrixRef(name=name))
#             seen.add(name)
#             if len(refs) >= limit:
#                 break

#     # If the heuristic finds nothing (can happen depending on ssgetpy metadata),
#     # fall back to the first `limit` problems so the pipeline is still runnable.
#     if not refs:
#         for m in mats[:limit]:
#             name = getattr(m, "name", None)
#             if name and name not in seen:
#                 refs.append(MatrixRef(name=name))
#                 seen.add(name)

#     return refs


def download_matrix(mat, data_dir: str = "data") -> str:
    """
    Download ssgetpy matrix in .mat format and return the path
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    mat_path = Path(os.path.join(data_dir, f"{mat.name}.mat"))
    if mat_path.exists():
        return str(mat_path)

    mat.download(format="MAT", destpath=str(data_dir))
    return str(mat_path)


def load_mat_csc(path: str, make_symmetric: bool = False, make_laplacian: bool = False) -> sp.csc_matrix:
    """
    Load .mat file and return CSC sparse matrix
    """

    mat = sio.loadmat(path)
    A = sp.csc_matrix(mat["Problem"]["A"][0, 0])

    if make_symmetric:
        A = _to_symmetric(A)
    if make_laplacian:
        A = _to_laplacian(A)

    return A


def save_mat_csc(mat, path: str):
    """
    Save .mat file to path
    """

    mdict = {
        "Problem": {
            "A": [[mat]]
        }
    }

    sio.savemat(path, mdict)


def rand_sparse_csc(
    n: int,
    density: float,
    symmetric: bool = False,
    spd: bool = False,
    seed: Optional[int] = None,
) -> sp.csc_matrix:
    """
    Generate a random sparse matrix in CSC format.
    """
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=density, format="csc", random_state=rng)

    if spd:
        A = A.T @ A + 1e-3 * sp.eye(n, format="csc")
    elif symmetric:
        A = _to_symmetric(A)

    return A.tocsc()    


def _to_symmetric(A):
    # return ((A + A.T) * 0.5).tocsc()
    A = A.tocsc()
    A = A + A.T
    A.data *= 0.5
    A.sum_duplicates()
    return A


def _to_laplacian(A, shift: float = 1e-3):
    """
    Build a shifted graph Laplacian from a sparse pattern.

    This produces an SPD matrix (after the diagonal shift) suitable for Cholesky.
    """

    A = A.tocsr(copy=False)

    # Use unweighted adjacency on the sparsity pattern, excluding diagonal.
    A = A.copy()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    A = A.maximum(A.T)

    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(deg, format="csr") - A
    if shift and shift > 0:
        L = L + shift * sp.eye(L.shape[0], format="csr")
    return L.tocsc()

def get_matrix(id: int):
    return ssgetpy.search(id)



def find_matrices(
    n: int,
    dtype: str = "real",
    rows: Optional[tuple] = None,
    cols: Optional[tuple] = None,
    nnzs: Optional[tuple] = None,
    spd: Optional[bool] = None,
    d23: Optional[bool] = None,
    group: Optional[str] = None,
    kind: Optional[str] = None,
    data_dir: str = "data",
) -> list[int]:

    mats = _ssget_search(
        limit=n,
        dtype=dtype,
        rows=rows,
        cols=cols,
        nnzs=nnzs,
        spd=spd,
        d23=d23,
        group=group,
        kind=kind
    )

    return [m.id for m in mats]

    # paths = []
    # for m in mats:
    #     paths.append(_download_matrix(m, data_dir=data_dir))

    # return paths

    # A = load_mat_as_csc(download_matrix_mat(name=ref.name, data_dir=data_dir))
    # if kind in ("spd_suitesparse", "symmetric_suitesparse", "curated_spd"):
    #     A = symmetrize_pattern(A)
    #     return A
    # elif kind == "laplacian":
    #     A = to_laplacian(A)
    #     return A
    # else:
    #     raise ValueError(f"Unknown kind: {kind}")
