"""Read-only, Mojo-accelerated subset of :mod:`pyBigWig`.

It reads standard little-endian local bigWig files and implements interval,
values and exact stats queries.  The public spelling intentionally follows
pyBigWig for applications that only need this query-oriented subset.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ._lib import bin_accumulate, values as kernel_values

__version__ = "0.1.0"
numpy = 1
remote = 0

_BW_MAGIC = 0x888FFC26
_CT_MAGIC = 0x78CA8C91
_RT_MAGIC = 0x2468ACE0


class BigWigError(RuntimeError):
    pass


def _at(buf: memoryview, offset: int, fmt: str):
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(buf):
        raise BigWigError("truncated bigWig file")
    return struct.unpack_from(fmt, buf, offset)


def _fixed_key(buf: memoryview, offset: int, size: int) -> str:
    if offset < 0 or size < 0 or offset + size > len(buf):
        raise BigWigError("truncated bigWig file")
    raw = bytes(buf[offset:offset + size]).split(b"\0", 1)[0]
    return raw.decode("utf-8")


def _slice(buf: memoryview, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0 or offset + size > len(buf):
        raise BigWigError("truncated bigWig file")
    return bytes(buf[offset:offset + size])


@dataclass(frozen=True)
class _IntervalData:
    starts: np.ndarray
    ends: np.ndarray
    values: np.ndarray


class bigWigFile:
    """An open local bigWig file, compatible with pyBigWig's read API subset."""

    def __init__(self, path: str | Path):
        self._path = str(path)
        self._raw = memoryview(Path(path).read_bytes())
        self._closed = False
        self._cached_query: tuple[tuple[int, int, int], _IntervalData] | None = None
        self._parse_header()
        self._chroms_by_id = self._read_chrom_tree()
        self._chroms = {name: size for name, (_, size) in self._chroms_by_id.items()}

    def _parse_header(self) -> None:
        (magic, version, zooms, chrom_tree, data, index, field_count, defined_count,
         auto_sql, total_summary, uncompress_size, reserved) = _at(self._raw, 0, "<IHHQQQHHQQIQ")
        if magic != _BW_MAGIC:
            raise BigWigError("not a little-endian bigWig file")
        if version < 3:
            raise BigWigError(f"unsupported bigWig version {version}")
        self._header = {"version": version, "nLevels": zooms, "nBasesCovered": 0,
                        "minVal": 0.0, "maxVal": 0.0, "sumData": 0.0, "sumSquared": 0.0}
        self._chrom_tree_offset = chrom_tree
        self._data_offset = data
        self._index_offset = index
        self._uncompress_size = uncompress_size
        self._total_summary_offset = total_summary
        if total_summary:
            bases, minimum, maximum, total, squared = _at(self._raw, total_summary, "<Qdddd")
            self._header.update(nBasesCovered=bases, minVal=minimum, maxVal=maximum,
                                sumData=total, sumSquared=squared)

    def _read_chrom_tree(self) -> dict[str, tuple[int, int]]:
        magic, block_size, key_size, val_size, item_count, reserved = _at(
            self._raw, self._chrom_tree_offset, "<IIIIQQ")
        if magic != _CT_MAGIC or val_size != 8:
            raise BigWigError("invalid chromosome B+ tree")
        found: dict[str, tuple[int, int]] = {}

        def visit(offset: int) -> None:
            leaf, _, count = _at(self._raw, offset, "<BBH")
            pos = offset + 4
            if leaf:
                for _ in range(count):
                    name = _fixed_key(self._raw, pos, key_size)
                    chrom_id, chrom_size = _at(self._raw, pos + key_size, "<II")
                    found[name] = (chrom_id, chrom_size)
                    pos += key_size + 8
            else:
                for _ in range(count):
                    visit(_at(self._raw, pos + key_size, "<Q")[0])
                    pos += key_size + 8

        visit(self._chrom_tree_offset + 32)
        if len(found) != item_count:
            raise BigWigError("incomplete chromosome B+ tree")
        return found

    def _block_locations(self, chrom_id: int, start: int, end: int) -> list[tuple[int, int]]:
        magic, block_size, item_count, *_ = _at(self._raw, self._index_offset, "<IIQIIIIQII")
        if magic != _RT_MAGIC:
            raise BigWigError("invalid circular index")
        blocks: list[tuple[int, int]] = []

        def overlaps(sc: int, sb: int, ec: int, eb: int) -> bool:
            return (ec > chrom_id or (ec == chrom_id and eb > start)) and (sc < chrom_id or (sc == chrom_id and sb < end))

        def visit(offset: int) -> None:
            leaf, _, count = _at(self._raw, offset, "<BBH")
            pos = offset + 4
            for _ in range(count):
                sc, sb, ec, eb = _at(self._raw, pos, "<IIII")
                if leaf:
                    data_offset, data_size = _at(self._raw, pos + 16, "<QQ")
                    if overlaps(sc, sb, ec, eb):
                        blocks.append((data_offset, data_size))
                    pos += 32
                else:
                    child = _at(self._raw, pos + 16, "<Q")[0]
                    if overlaps(sc, sb, ec, eb):
                        visit(child)
                    pos += 24

        visit(self._index_offset + 48)
        return blocks

    def _decode(self, chrom_id: int, start: int, end: int) -> _IntervalData:
        starts: list[int] = []
        ends: list[int] = []
        values: list[float] = []
        for offset, size in self._block_locations(chrom_id, start, end):
            payload = _slice(self._raw, offset, size)
            if self._uncompress_size:
                try:
                    payload = zlib.decompress(payload)
                except zlib.error as exc:
                    raise BigWigError("invalid compressed bigWig block") from exc
            block = memoryview(payload)
            chrom, section_start, section_end, step, span, section_type, _, count = _at(
                block, 0, "<IIIIIBBH")
            if chrom != chrom_id:
                continue
            pos = 24
            if section_type == 1:
                for _ in range(count):
                    lo, hi, value = _at(block, pos, "<IIf")
                    if hi > start and lo < end:
                        starts.append(lo); ends.append(hi); values.append(value)
                    pos += 12
            elif section_type == 2:
                for _ in range(count):
                    lo, value = _at(block, pos, "<If")
                    hi = lo + span
                    if hi > start and lo < end:
                        starts.append(lo); ends.append(hi); values.append(value)
                    pos += 8
            elif section_type == 3:
                for index in range(count):
                    value = _at(block, pos, "<f")[0]
                    lo = section_start + index * step
                    hi = lo + span
                    if hi > start and lo < end:
                        starts.append(lo); ends.append(hi); values.append(value)
                    pos += 4
            else:
                raise BigWigError(f"unsupported bigWig section type {section_type}")
        order = np.argsort(starts, kind="stable") if starts else np.empty(0, dtype=int)
        return _IntervalData(np.asarray(starts, dtype=np.int64)[order],
                             np.asarray(ends, dtype=np.int64)[order],
                             np.asarray(values, dtype=np.float64)[order])

    def _query(self, chrom: str, start: int | None, end: int | None) -> tuple[int, int, _IntervalData]:
        self._check_open()
        if chrom not in self._chroms_by_id:
            raise RuntimeError(f"Invalid chromosome name: {chrom}")
        chrom_id, chrom_size = self._chroms_by_id[chrom]
        start = 0 if start is None else start
        end = chrom_size if end is None else end
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start or end > chrom_size:
            raise RuntimeError("Invalid interval bounds")
        key = (chrom_id, start, end)
        if self._cached_query is not None and self._cached_query[0] == key:
            data = self._cached_query[1]
        else:
            data = self._decode(chrom_id, start, end)
            self._cached_query = (key, data)
        return start, end, data

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("The bigWig file is closed")

    def chroms(self, chrom: str | None = None):
        self._check_open()
        return self._chroms.get(chrom) if chrom is not None else dict(self._chroms)

    def header(self):
        self._check_open()
        return dict(self._header)

    def intervals(self, chrom: str, start: int | None = None, end: int | None = None):
        _, _, data = self._query(chrom, start, end)
        if data.starts.size == 0:
            return None
        return [(int(lo), int(hi), float(value)) for lo, hi, value in zip(data.starts, data.ends, data.values)]

    def values(self, chrom: str, start: int, end: int, numpy: bool = False):
        start, end, data = self._query(chrom, start, end)
        result = np.full(end - start, np.nan, dtype=np.float64)
        if data.starts.size:
            kernel_values(data.starts, data.ends, data.values, start, end, result)
        return result if numpy else result.tolist()

    def stats(self, chrom: str, start: int | None = None, end: int | None = None,
              type: Literal["mean", "min", "max", "coverage", "std", "sum"] = "mean",
              nBins: int = 1, exact: bool = False):
        if type not in {"mean", "min", "max", "coverage", "std", "sum"}:
            raise RuntimeError(f"Invalid type: {type}")
        if not isinstance(nBins, int) or nBins < 1:
            raise RuntimeError("nBins must be a positive integer")
        start, end, data = self._query(chrom, start, end)
        if end == start:
            return [None] * nBins
        acc = np.empty((nBins, 5), dtype=np.float64)
        if data.starts.size:
            bin_accumulate(data.starts, data.ends, data.values, start, end, nBins, acc)
        else:
            acc.fill(0.0)
        covered = acc[:, 1]
        result: list[float | None] = []
        for index, cov in enumerate(covered):
            if cov == 0:
                result.append(None)
                continue
            if type == "mean": value = acc[index, 0] / cov
            elif type == "min": value = acc[index, 2]
            elif type == "max": value = acc[index, 3]
            elif type == "coverage":
                bin_start = start + (index * (end - start)) // nBins
                bin_end = start + ((index + 1) * (end - start)) // nBins
                value = cov / (bin_end - bin_start) if bin_end > bin_start else 0.0
            elif type == "sum": value = acc[index, 0]
            else:
                mean = acc[index, 0] / cov
                value = (max(0.0, (acc[index, 4] - acc[index, 0] * mean) / (cov - 1)) ** 0.5
                         if cov > 1 else 0.0)
            result.append(float(value))
        return result

    def close(self) -> None:
        self._closed = True
        self._cached_query = None
        self._raw.release()

    def isBigWig(self) -> bool:
        return not self._closed

    def isBigBed(self) -> bool:
        return False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open(file: str | Path, mode: str = "r") -> bigWigFile:
    """Open a local bigWig file for reading. Writing and remote URLs are unsupported."""
    if mode not in {"r", "rb"}:
        raise RuntimeError("mojo-pybigwig currently supports read-only mode")
    return bigWigFile(file)
