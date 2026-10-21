"""Measures where the GPU module starts to pay off, to choose kDefaultGpuMinNodes.

Generates VGA grids of increasing size from a drawing (VISPREP), times the visual local
and global measures on all CPU cores and on the GPU (-gpu -gpumin 0), and reports the
smallest map from which the GPU is faster for every larger map too. Run it on the
target machine (e.g. an Apple Silicon Mac with the Metal module) and put the result in
gpu/host/accelcommon.h (kDefaultGpuMinNodes, per platform).

Each timing is the best of --repeat runs of the analysis itself (the -t times file), so
loading and saving the file are excluded; the GPU time includes the module's set-up
(Metal compiles its kernels when first used in a process).

  python3 tools/mac/bench_gpu_threshold.py --cli build/depthmapXcli/depthmapXcli \\
      --drawing ../depthmapX-rapido-2026/exemplos/galeria_vazia.graph --fill 0.68,7.04 \\
      --out /tmp/dmx-bench
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpu"))
from validate import analysis_seconds  # noqa: E402

MEASURES = {"local": ["-vl"], "global": ["-vg", "-vr", "n"]}


def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def points_of(cli, graph, out_dir):
    csv = os.path.join(out_dir, "points.csv")
    run([cli, "-m", "EXPORT", "-f", graph, "-o", csv, "-em", "pointmap-data-csv"])
    with open(csv, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--drawing", required=True, help=".graph with the drawing layers (no grid)")
    ap.add_argument("--fill", required=True, help="x,y of a point inside the space to fill")
    ap.add_argument("--spacings", default="0.12,0.09,0.07,0.055,0.045,0.04,0.035,0.03,0.025,0.02,0.015",
                    help="grid spacings, coarse to fine (drawing units)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for spacing in a.spacings.split(","):
        graph = os.path.join(a.out, f"grid_{spacing}.graph")
        if not os.path.exists(graph):
            run([a.cli, "-m", "VISPREP", "-f", a.drawing, "-o", graph, "-pg", spacing, "-pp", a.fill, "-pm"])
        n = points_of(a.cli, graph, a.out)
        row = {"spacing": spacing, "points": n}
        for name, flags in MEASURES.items():
            for mode, extra in (("cpu", []), ("gpu", ["-gpu", "-gpumin", "0"])):
                best = float("inf")
                for _ in range(a.repeat):
                    times = os.path.join(a.out, "times.csv")
                    out = run([a.cli, "-m", "VGA", "-vm", "visibility", *flags, *extra, "-f", graph,
                               "-o", os.path.join(a.out, "result.graph"), "-t", times])
                    if mode == "gpu" and "GPU (" not in out:
                        sys.exit("the GPU was not used:\n" + out)
                    best = min(best, analysis_seconds(times))
                row[f"{name}_{mode}"] = best
        rows.append(row)
        print(f"{n:8d} points  local cpu {row['local_cpu']:8.3f} s gpu {row['local_gpu']:8.3f} s   "
              f"global cpu {row['global_cpu']:8.3f} s gpu {row['global_gpu']:8.3f} s", flush=True)

    print("\n| points | local CPU (s) | local GPU (s) | global CPU (s) | global GPU (s) |")
    print("|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['points']} | {r['local_cpu']:.3f} | {r['local_gpu']:.3f} | {r['global_cpu']:.3f} | "
              f"{r['global_gpu']:.3f} |")
    for name in MEASURES:
        # smallest map from which the GPU is faster for this and every larger map
        threshold = None
        for i, r in enumerate(rows):
            if all(x[f"{name}_gpu"] < x[f"{name}_cpu"] for x in rows[i:]):
                threshold = r["points"]
                break
        print(f"{name}: GPU faster from {threshold} points" if threshold else
              f"{name}: GPU never faster on every larger map tested")


if __name__ == "__main__":
    main()
