# mojo-pybigwig

`mojo-pybigwig` is a read-only local bigWig interval-query port with the
query-facing names and signatures of [pyBigWig](https://github.com/deeptools/pyBigWig).
It reads indexed, zlib-compressed little-endian bigWig files itself and runs the
hot interval expansion and exact binned-statistics loops in Mojo.

## Covered subset

`open(path, "r")`, `bigWigFile.chroms`, `header`, `intervals`, `values`,
`stats`, `close`, `isBigWig`, `isBigBed`, and context-manager use are covered.
`values` preserves pyBigWig's list-with-`nan` or NumPy-array behavior; exact
`mean`, `min`, `max`, `coverage`, `std`, and `sum` statistics are supported.
The `exact` argument is accepted; this reader always uses the exact path.

Not covered: writing, bigBed, remote URLs, zoom-level/approximate statistics,
HTTP caching, and pyBigWig's optional NumPy/C-extension configuration knobs.

## Install and use

```bash
pixi install
pixi run build
pixi run python -c 'import pyBigWig; bw = pyBigWig.open("signal.bw"); print(bw.chroms()); print(bw.values("chr1", 0, 100, numpy=True)); bw.close()'
```

Replace `signal.bw` with a local little-endian bigWig and select an existing
chromosome and interval.

## How it works

The Python layer parses the bigWig header, chromosome B+ tree, R-tree block
index and bedGraph/variableStep/fixedStep sections. It hands contiguous
double-precision value buffers and signed-integer coordinate buffers to one
Mojo shared library through ctypes addresses. Mojo owns no memory: Python owns
the NumPy buffers for the full native call, and validates their type, layout,
length, and writability before invoking either kernel.

## Benchmark

Measured on this machine with `pixi run bench`. The benchmark uses the same
locally generated file, excludes one-time Mojo compilation, and reports the
best of its repeated runs.

| kernel | mojo-pybigwig | pyBigWig | speedup |
| --- | ---: | ---: | ---: |
| values 5M bases | 4.92 ms | 26.53 ms | 5.39x |
| exact mean 10k bins | 16.55 ms | 1624.58 ms | 98.15x |

The reader retains the last decoded query, avoiding repeated decompression and
array construction for repeated interval requests. `values` allocates an
uninitialized NumPy result and uses one Mojo pass to SIMD-fill both covered runs
and NaN gaps. Large non-overlapping queries are split into disjoint coordinate
regions after a size threshold; smaller and overlapping queries stay serial.
Exact statistics remains a single Mojo overlap-accumulation call.

There is no GPU path. Interval expansion is a bandwidth-bound store kernel, and
exact bin accumulation is also below the roughly two-flops-per-byte threshold;
host/device transfers and launch overhead would make either path slower. Run
`pixi run bench` to measure your machine.

## Verification

`pixi run test` creates bigWigs using upstream pyBigWig and asserts every
documented query method, list and NumPy values, all supported statistics, and
all supported section layouts against that upstream reader.

## License

MIT
