import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from cholesky_bench import run_bench
from matrices import download_matrix, find_matrices, get_matrix, load_mat_csc, rand_sparse_csc
import numpy as np
import math


def set_env_threads(nthreads: int):
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)


def run_one(size: int, density: float, args: argparse.Namespace) -> list[dict]:
    """
    Runs the benchmark for one matrix.
    """

    A = rand_sparse_csc(size, density, symmetric=True, spd=True)

    out = []
    for o in args.orderings.split(","):
        if o.strip().lower() not in ("natural", "amd", "metis", "nesdis"):
            raise ValueError(f"Unknown ordering: {o.strip()}")
        res = run_bench(
            A, ordering=o.strip().lower(), nthreads=args.nthreads, repeats=args.repeats
        )
        res["matrix_id"] = time.time()
        res["matrix_name"] = time.time()
        res["group"] = "custom"
        res["matrix_kind"] = "random sparse spd"
        res["pattern_sym"] = 1.0
        res["numeric_sym"] = 1.0
        res["nrows"] = size
        res["ncols"] = size
        res["nnz"] = A.nnz
        res["dtype"] = 'real'
        res["2d3d"] = False
        res["spd"] = True
        out.append(res)

    return out


def run(args: argparse.Namespace):
    """
    Runs the benchmark
    """
    set_env_threads(args.nthreads)

    run_id = time.time()
    out = Path(os.path.join(args.out_dir, f"results_{run_id}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    min_base, max_base = math.log10(args.min_size), math.log10(args.max_size)
    pbar = tqdm(np.logspace(min_base, max_base, num=args.nmats), desc="Running benchmarks", total=args.nmats)
    for s in pbar:

        density = float(np.random.rand(1)[0])
        density = (density * (math.log10(args.max_density) - math.log10(args.min_density))) + math.log10(args.min_density)
        density = 10.0**density

        pbar.set_postfix({"size": s, "density": density})
        res = run_one(int(s), density, args)
        results.extend(res)
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--orderings", required=True)
    ap.add_argument("--nthreads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=10)  # num runs per matrix
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--use-cache", type=bool, default=True)
    ap.add_argument("--max-size", type=int)
    ap.add_argument("--min-size", type=int)
    ap.add_argument("--max-density", type=float)
    ap.add_argument("--min-density", type=float)
    ap.add_argument("--nmats", type=int, default=100)  # num matrices
    args = ap.parse_args()

    # set in run_bench
    # set_env_threads(args.nthreads)

    run(args)
