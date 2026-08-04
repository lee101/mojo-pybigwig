"""Interval-query benchmark against upstream pyBigWig on the same file."""

import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))
import pyBigWig as mojo_bw
from pyBigWig._lib import lib


def upstream_module():
    search = [p for p in sys.path if Path(p).resolve() != (ROOT / "python").resolve()]
    spec = importlib.machinery.PathFinder.find_spec("pyBigWig", search)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timed(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        then = time.perf_counter(); fn(); best = min(best, time.perf_counter() - then)
    return best


def main():
    upstream = upstream_module()
    path = Path(tempfile.gettempdir()) / "mojo-pybigwig-bench.bw"
    n = 250_000
    starts = np.arange(n, dtype=np.int64) * 20
    writer = upstream.open(str(path), "w")
    writer.addHeader([("chr1", int(starts[-1] + 20))])
    writer.addEntries(["chr1"] * n, starts.tolist(), ends=(starts + 20).tolist(),
                      values=np.sin(starts / 10000).astype(float).tolist())
    writer.close()
    mine, ref = mojo_bw.open(path), upstream.open(str(path))
    # Loading the shared library can compile it on first use; exclude that setup cost.
    lib()
    cases = [("values 5M bases", lambda r: r.values("chr1", 0, int(starts[-1] + 20), numpy=True)),
             ("exact mean 10k bins", lambda r: r.stats("chr1", 0, int(starts[-1] + 20), nBins=10_000, exact=True))]
    print("| kernel | mojo-pybigwig | pyBigWig | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, query in cases:
        mojo_s, ref_s = timed(lambda: query(mine)), timed(lambda: query(ref))
        print(f"| {name} | {mojo_s * 1000:.2f} ms | {ref_s * 1000:.2f} ms | {ref_s / mojo_s:.2f}x |")
    mine.close(); ref.close()


if __name__ == "__main__":
    main()
