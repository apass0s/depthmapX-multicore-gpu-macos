"""Compare two depthmapX attribute CSV exports column by column.

Usage: python compare_outputs.py reference.csv candidate.csv [--rtol 1e-5] [--atol 1e-6] [--key Ref]

Rows are matched on the key column (default: Ref, falling back to the first column).
Every column present in both files is compared; columns present in only one file are
reported. Exit code is 0 when all shared columns match within tolerance, 1 otherwise.
"""
import argparse
import csv
import math
import sys


def load(path, key):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}, []
    cols = list(rows[0].keys())
    if key not in cols:
        key = cols[0]
    return {r[key]: r for r in rows}, cols


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--key", default="Ref")
    a = ap.parse_args()

    ref, rcols = load(a.reference, a.key)
    cand, ccols = load(a.candidate, a.key)
    ok = True

    only_ref = [c for c in rcols if c not in ccols]
    only_cand = [c for c in ccols if c not in rcols]
    if only_ref:
        print(f"columns only in reference: {only_ref}")
        ok = False
    if only_cand:
        print(f"columns only in candidate: {only_cand}")
        ok = False
    missing = set(ref) ^ set(cand)
    if missing:
        print(f"{len(missing)} row keys not present in both files")
        ok = False

    keys = [k for k in ref if k in cand]
    print(f"{len(keys)} rows compared, tolerance: |a-b| <= {a.atol} + {a.rtol}*|a|")
    print(f"{'column':45} {'mismatch':>9} {'max abs diff':>14} {'max rel diff':>14}")
    for c in (c for c in rcols if c in ccols):
        bad = 0
        max_abs = max_rel = 0.0
        for k in keys:
            x, y = ref[k][c], cand[k][c]
            fx, fy = num(x), num(y)
            if fx is None or fy is None:
                if x != y:
                    bad += 1
                continue
            if math.isnan(fx) and math.isnan(fy):
                continue
            d = abs(fx - fy)
            rel = d / abs(fx) if fx != 0 else (0.0 if d == 0 else math.inf)
            max_abs, max_rel = max(max_abs, d), max(max_rel, rel)
            if d > a.atol + a.rtol * abs(fx):
                bad += 1
        flag = "" if bad == 0 else "  <-- DIFFERS"
        print(f"{c[:45]:45} {bad:9d} {max_abs:14.6g} {max_rel:14.6g}{flag}")
        ok = ok and bad == 0
    print("RESULT:", "MATCH" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
