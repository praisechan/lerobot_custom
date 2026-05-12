#!/usr/bin/env python3
"""
Compute overlap between the streaming and WMMA kernels from an exported Nsight
Systems SQLite database.

Usage:
    python3 tools/nsys_overlap.py build/nsys_short.sqlite
"""

import sqlite3
import sys


MEM_PREFIX = "streaming_triad_kernel"
COMPUTE_PREFIX = "wmma_compute_kernel"


def load_intervals(path):
    con = sqlite3.connect(path)
    rows = con.execute(
        """
        select k.start, k.end, s.value, k.greenContextId, k.streamId
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on k.demangledName = s.id
        where s.value like ? or s.value like ?
        order by k.start
        """,
        (MEM_PREFIX + "%", COMPUTE_PREFIX + "%"),
    ).fetchall()
    con.close()
    return rows


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    rows = load_intervals(sys.argv[1])
    mem = [r for r in rows if r[2].startswith(MEM_PREFIX)]
    compute = [r for r in rows if r[2].startswith(COMPUTE_PREFIX)]

    overlaps = []
    for m in mem:
        for c in compute:
            overlap = max(0, min(m[1], c[1]) - max(m[0], c[0]))
            if overlap:
                overlaps.append((overlap, m, c))

    total_overlap_ms = sum(ov for ov, _, _ in overlaps) / 1e6
    print(f"streaming kernels: {len(mem)}")
    print(f"wmma kernels:      {len(compute)}")
    print(f"overlap pairs:     {len(overlaps)}")
    print(f"total pairwise overlap: {total_overlap_ms:.3f} ms")

    for overlap, m, c in overlaps:
        print(
            f"  {overlap / 1e6:.3f} ms: "
            f"mem ctx={m[3]} stream={m[4]} [{m[0] / 1e6:.3f}, {m[1] / 1e6:.3f}] "
            f"with compute ctx={c[3]} stream={c[4]} [{c[0] / 1e6:.3f}, {c[1] / 1e6:.3f}]"
        )


if __name__ == "__main__":
    main()
