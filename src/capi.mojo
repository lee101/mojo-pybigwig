"""Hot interval-query loops exposed to the Python bigWig reader."""

from std.sys.info import simd_width_of

comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


@export("mpbw_values")
def mpbw_values(starts_addr: Int, ends_addr: Int, values_addr: Int, n: Int,
                query_start: Int, query_end: Int, result_addr: Int) abi("C"):
    """Write each covered base of a query to a pre-filled float64 result."""
    # Python validates the addresses and backing array lengths before this ABI call.
    # Keep this guard before creating pointers so a no-op call needs no valid address.
    if n <= 0 or query_end <= query_start:
        return
    var starts = IPtr(unsafe_from_address=starts_addr)
    var ends = IPtr(unsafe_from_address=ends_addr)
    var values = FPtr(unsafe_from_address=values_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    var i = 0
    while i < n:
        var lo = max(starts.load(i), query_start)
        var hi = min(ends.load(i), query_end)
        if lo < hi:
            var pos = lo
            var value = values.load(i)
            var vector_end = hi - ((hi - pos) % W)
            var packed_value = SIMD[DType.float64, W](value)
            while pos < vector_end:
                result.store(pos - query_start, packed_value)
                pos += W
            while pos < hi:
                result.store(pos - query_start, value)
                pos += 1
        i += 1


@export("mpbw_bin_accumulate")
def mpbw_bin_accumulate(starts_addr: Int, ends_addr: Int, values_addr: Int, n: Int,
                        query_start: Int, query_end: Int, bins: Int,
                        acc_addr: Int) abi("C"):
    """Accumulate sum, coverage, min, max and sum-of-squares for exact bins."""
    # See mpbw_values: all non-empty calls are validated by the Python wrapper.
    if n <= 0 or bins <= 0 or query_end <= query_start:
        return
    var starts = IPtr(unsafe_from_address=starts_addr)
    var ends = IPtr(unsafe_from_address=ends_addr)
    var values = FPtr(unsafe_from_address=values_addr)
    var acc = FPtr(unsafe_from_address=acc_addr)
    var length = query_end - query_start
    var b = 0
    while b < bins:
        var offset = b * 5
        acc.store(offset, 0.0)
        acc.store(offset + 1, 0.0)
        # bigWig payload values are finite float32 values, whose range is below 3.5e38.
        acc.store(offset + 2, 3.5e38)
        acc.store(offset + 3, -3.5e38)
        acc.store(offset + 4, 0.0)
        b += 1

    var i = 0
    while i < n:
        var lo = max(starts.load(i), query_start)
        var hi = min(ends.load(i), query_end)
        if lo < hi:
            var first_bin = ((lo - query_start) * bins) // length
            var last_bin = ((hi - 1 - query_start) * bins) // length
            var value = values.load(i)
            var bin = first_bin
            while bin <= last_bin:
                var bin_start = query_start + (bin * length) // bins
                var bin_end = query_start + ((bin + 1) * length) // bins
                var covered_start = max(lo, bin_start)
                var covered_end = min(hi, bin_end)
                var weight = Float64(covered_end - covered_start)
                var offset = bin * 5
                acc.store(offset, acc.load(offset) + value * weight)
                acc.store(offset + 1, acc.load(offset + 1) + weight)
                if value < acc.load(offset + 2):
                    acc.store(offset + 2, value)
                if value > acc.load(offset + 3):
                    acc.store(offset + 3, value)
                acc.store(offset + 4, acc.load(offset + 4) + value * value * weight)
                bin += 1
        i += 1
