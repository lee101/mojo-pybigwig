import math
from pathlib import Path

import numpy as np
import pyBigWig as mojo_bw
import pytest


@pytest.fixture()
def fixture(tmp_path: Path):
    # Resolve the installed extension before this test's PYTHONPATH shadowing.
    import importlib.machinery
    import importlib.util
    import sys
    root = Path(__file__).parents[1] / "python"
    search = [p for p in sys.path if Path(p).resolve() != root.resolve()]
    spec = importlib.machinery.PathFinder.find_spec("pyBigWig", search)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "fixture.bw"
    writer = module.open(str(path), "w")
    writer.addHeader([("chr1", 1000), ("chr2", 100)])
    writer.addEntries(["chr1"] * 5 + ["chr2"], [0, 13, 100, 250, 990, 10],
                      ends=[10, 37, 180, 400, 1000, 20], values=[1.5, -2.0, 4.25, 0.5, 8.0, 3.0])
    writer.close()
    return path, module


def test_metadata_and_intervals_match_upstream(fixture):
    path, upstream = fixture
    mine, ref = mojo_bw.open(path), upstream.open(str(path))
    assert mine.chroms() == ref.chroms()
    assert mine.chroms("chr1") == ref.chroms("chr1")
    assert mine.intervals("chr1") == list(ref.intervals("chr1"))
    assert mine.intervals("chr1", 15, 260) == list(ref.intervals("chr1", 15, 260))
    assert mine.intervals("chr2", 0, 5) is None
    assert mine.header() == pytest.approx(ref.header())
    mine.close(); ref.close()


@pytest.mark.parametrize("start,end", [(0, 50), (5, 271), (900, 1000)])
def test_values_match_upstream_lists_and_numpy(fixture, start, end):
    path, upstream = fixture
    mine, ref = mojo_bw.open(path), upstream.open(str(path))
    got, expected = mine.values("chr1", start, end), ref.values("chr1", start, end)
    assert len(got) == len(expected)
    for actual, wanted in zip(got, expected):
        assert math.isnan(actual) if math.isnan(wanted) else actual == pytest.approx(wanted)
    np.testing.assert_allclose(mine.values("chr1", start, end, numpy=True),
                               ref.values("chr1", start, end, numpy=True), equal_nan=True)
    mine.close(); ref.close()


def test_values_simd_tail_matches_upstream(fixture):
    path, upstream = fixture
    mine, ref = mojo_bw.open(path), upstream.open(str(path))
    np.testing.assert_allclose(mine.values("chr1", 3, 18, numpy=True),
                               ref.values("chr1", 3, 18, numpy=True), equal_nan=True)
    mine.close(); ref.close()


def test_values_kernel_fills_gaps_overlaps_and_simd_tails():
    from pyBigWig._lib import values

    starts = np.array([3, 11, 14], dtype=np.int64)
    ends = np.array([8, 17, 19], dtype=np.int64)
    payload = np.array([1.25, 2.5, -4.0], dtype=np.float64)
    result = np.empty(23, dtype=np.float64)
    values(starts, ends, payload, 0, 23, result)
    expected = np.full(23, np.nan, dtype=np.float64)
    expected[3:8] = 1.25
    expected[11:17] = 2.5
    expected[14:19] = -4.0
    np.testing.assert_array_equal(result, expected)


def test_values_kernel_parallel_threshold_and_tail():
    from pyBigWig._lib import values

    n = 16_385
    starts = np.arange(n, dtype=np.int64) * 65
    ends = starts + 33
    payload = np.linspace(-3.0, 7.0, n, dtype=np.float64)
    length = 1_064_997
    result = np.empty(length, dtype=np.float64)
    values(starts, ends, payload, 0, length, result)
    for index in (0, 1, n // 2, n - 1):
        assert np.all(result[starts[index]:ends[index]] == payload[index])
        if ends[index] < length:
            assert math.isnan(result[ends[index]])
    assert math.isnan(result[-1])


def test_reuses_the_last_decoded_query(fixture):
    path, _ = fixture
    mine = mojo_bw.open(path)
    original = mine._decode
    calls = 0

    def counting_decode(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    mine._decode = counting_decode
    mine.values("chr1", 3, 18, numpy=True)
    mine.values("chr1", 3, 18, numpy=True)
    assert calls == 1
    mine.close()


@pytest.mark.parametrize("kind", ["mean", "min", "max", "coverage", "std", "sum"])
@pytest.mark.parametrize("bins", [1, 7, 29])
def test_exact_stats_match_upstream(fixture, kind, bins):
    path, upstream = fixture
    mine, ref = mojo_bw.open(path), upstream.open(str(path))
    got = mine.stats("chr1", 0, 500, type=kind, nBins=bins, exact=True)
    expected = ref.stats("chr1", 0, 500, type=kind, nBins=bins, exact=True)
    assert got == pytest.approx(expected, nan_ok=True)
    mine.close(); ref.close()


def test_context_manager_and_errors(fixture):
    path, _ = fixture
    with mojo_bw.open(path) as reader:
        assert reader.isBigWig() and not reader.isBigBed()
        with pytest.raises(RuntimeError): reader.values("missing", 0, 1)
        with pytest.raises(RuntimeError): reader.stats("chr1", 1, 0)
    with pytest.raises(RuntimeError): reader.chroms()


def test_open_modes_and_default_exact_path_match_upstream(fixture):
    path, upstream = fixture
    mine, ref = mojo_bw.open(path, "rb"), upstream.open(str(path))
    assert mine.stats("chr1", 0, 500, type="mean", nBins=7) == pytest.approx(
        ref.stats("chr1", 0, 500, type="mean", nBins=7, exact=True), nan_ok=True
    )
    with pytest.raises(RuntimeError):
        mojo_bw.open(path, "w")
    mine.close(); ref.close()


@pytest.mark.parametrize("layout", ["variable", "fixed"])
def test_variable_and_fixed_step_sections_match_upstream(tmp_path, layout):
    import importlib.machinery
    import importlib.util
    import sys
    root = Path(__file__).parents[1] / "python"
    spec = importlib.machinery.PathFinder.find_spec(
        "pyBigWig", [p for p in sys.path if Path(p).resolve() != root.resolve()]
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    path = tmp_path / f"{layout}.bw"
    writer = upstream.open(str(path), "w")
    writer.addHeader([("chr1", 100)])
    if layout == "variable":
        writer.addEntries("chr1", [0, 12, 40], values=[1.0, -3.0, 2.5], span=5)
    else:
        writer.addEntries("chr1", 0, values=[1.0, -3.0, 2.5], span=5, step=12)
    writer.close()
    mine, ref = mojo_bw.open(path), upstream.open(str(path))
    assert mine.intervals("chr1") == list(ref.intervals("chr1"))
    np.testing.assert_allclose(mine.values("chr1", 0, 50, numpy=True),
                               ref.values("chr1", 0, 50, numpy=True), equal_nan=True)
    mine.close(); ref.close()
