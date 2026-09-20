#!/usr/bin/env python3
"""Host-only PTQ1 trit codec probe.

This checks the packed byte arithmetic and the 128-element position map used by
the PTQ1 CPU codec. It does not compile or execute a device kernel.
"""

from __future__ import annotations

import argparse
import sys


QK = 128
N_QS = 24
N_QH = 2
POW3 = (1, 3, 9, 27, 81)


def position_map() -> list[tuple[str, int, int]]:
    """Translate the CPU stage loops into (field, byte, trit-plane) entries."""
    result: list[tuple[str, int, int]] = []
    byte = 0
    for width in (32, 16, 8):
        while byte + width <= N_QS:
            for plane in range(5):
                for offset in range(width):
                    result.append(("qs", byte + offset, plane))
            byte += width
    for plane in range(4):
        for offset in range(N_QH):
            result.append(("qh", offset, plane))
    if len(result) != QK:
        raise AssertionError(f"position map has {len(result)} entries")
    return result


def decode_trit(byte: int, plane: int, wrap_u8: bool) -> int:
    value = byte * POW3[plane]
    if wrap_u8:
        value &= 0xFF
    return (value * 3 >> 8) - 1



def closed_form_position(pos: int) -> tuple[str, int, int]:
    if pos < 80:
        return ("qs", pos % 16, pos // 16)
    if pos < 120:
        q = pos - 80
        return ("qs", 16 + q % 8, q // 8)
    q = pos - 120
    return ("qh", q % 2, q // 2)

def check_arithmetic() -> tuple[int, int, tuple[int, int, int, int]]:
    mismatches = 0
    first = None
    for byte in range(256):
        for plane in range(5):
            expected = decode_trit(byte, plane, True)
            faulty = decode_trit(byte, plane, False)
            if expected != faulty:
                mismatches += 1
                if first is None:
                    first = (byte, plane, expected, faulty)
    if first is None:
        raise AssertionError("faulty arithmetic unexpectedly matched")
    return mismatches, 256 * len(POW3), first


def check_positions() -> int:
    mapped = position_map()
    for pos, entry in enumerate(mapped):
        expected = closed_form_position(pos)
        if entry != expected:
            raise AssertionError(f"position map mismatch at {pos}: {entry} != {expected}")
    qs = tuple((37 * i + 11) & 0xFF for i in range(N_QS))
    qh = tuple((83 * i + 19) & 0xFF for i in range(N_QH))
    decoded = []
    for field, byte, plane in mapped:
        packed = qs[byte] if field == "qs" else qh[byte]
        decoded.append(decode_trit(packed, plane, True))
    if len(decoded) != QK:
        raise AssertionError("decoded position count is not 128")
    if not all(-1 <= value <= 1 for value in decoded):
        raise AssertionError("decoded trit is outside {-1, 0, 1}")
    return len(decoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-fault",
        action="store_true",
        help="also print the expected rejection of arithmetic without the u8 wrap",
    )
    args = parser.parse_args()

    mismatches, cases, first = check_arithmetic()
    positions = check_positions()
    print(f"PTQ1 codec probe: PASS ({positions} positions, {cases} byte/plane cases)")
    print(
        "required u8 wrap: PASS "
        f"(faulty variant rejected: {mismatches} mismatches; "
        f"first byte={first[0]}, plane={first[1]}, expected={first[2]}, faulty={first[3]})"
    )
    if args.demo_fault:
        print("intentional faulty arithmetic: REJECTED")
    print("limit: host arithmetic and position mapping only; no GPU kernel proof")
    return 0


if __name__ == "__main__":
    sys.exit(main())
