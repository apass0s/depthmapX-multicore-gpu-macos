"""Validate and time the accelerated analyses against a reference.

Runs each analysis case with several depthmapXcli builds/modes, exports every
attribute to CSV, compares each run with the reference run column by column
(compare_outputs.py) and prints a timing table.

Example:
  python tools/gpu/validate.py --cli C:/dmx/build/depthmapXcli/depthmapXcli.exe \
      --rapido ../depthmapX-rapido-2026/motor/depthmapXcli.exe \
      --graph ../depthmapX-rapido-2026/exemplos/galeria_grade_0.04.graph \
      --modes serial,cpu,gpu,rapido --reference serial --cases vis_local --out C:/dmx/validate

Modes: serial (new build, -serial = original upstream code), cpu (new build,
multicore), gpu (new build, -gpu -gpumin 0), rapido (the rapido build, if given).
The GPU mode fails validation if the run report does not say the GPU was used.
Exit code 0 only if every comparison matches.
"""
import argparse
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_outputs  # noqa: E402

# name -> (CLI arguments, export mode, rtol, atol)
CASES = {
    "vis_local": (["-m", "VGA", "-vm", "visibility", "-vl"], "pointmap-data-csv", 0.0, 0.0),
    "vis_global": (["-m", "VGA", "-vm", "visibility", "-vg", "-vr", "n"], "pointmap-data-csv", 0.0, 0.0),
    "vis_global_r3": (["-m", "VGA", "-vm", "visibility", "-vg", "-vr", "3"], "pointmap-data-csv", 0.0, 0.0),
    "metric": (["-m", "VGA", "-vm", "metric", "-vr", "n"], "pointmap-data-csv", 0.0, 0.0),
    "metric_r": (["-m", "VGA", "-vm", "metric", "-vr", "1.5"], "pointmap-data-csv", 0.0, 0.0),
    "angular": (["-m", "VGA", "-vm", "angular"], "pointmap-data-csv", 0.0, 0.0),
    "seg_angular": (["-m", "SEGMENT", "-st", "angular", "-sr", "n"], "shapegraph-map-csv", 0.0, 0.0),
    "seg_angular_r": (["-m", "SEGMENT", "-st", "angular", "-sr", "1,2,n"], "shapegraph-map-csv", 0.0, 0.0),
    "seg_tulip": (["-m", "SEGMENT", "-st", "tulip", "-sr", "n", "-srt", "metric", "-sic", "-stb", "1024"],
                  "shapegraph-map-csv", 0.0, 0.0),
    "seg_tulip_r": (["-m", "SEGMENT", "-st", "tulip", "-sr", "250,n", "-srt", "metric", "-sic", "-stb", "8"],
                    "shapegraph-map-csv", 0.0, 0.0),
    "seg_tulip_w": (["-m", "SEGMENT", "-st", "tulip", "-sr", "n", "-srt", "metric", "-sic", "-stb", "1024",
                     "-swa", "Segment Length"], "shapegraph-map-csv", 0.0, 0.0),
    "seg_tulip_steps": (["-m", "SEGMENT", "-st", "tulip", "-sr", "3,n", "-srt", "steps", "-sic", "-stb", "64"],
                        "shapegraph-map-csv", 0.0, 0.0),
    "seg_tulip_ang": (["-m", "SEGMENT", "-st", "tulip", "-sr", "1,n", "-srt", "angular", "-stb", "1024"],
                      "shapegraph-map-csv", 0.0, 0.0),
    "seg_topological": (["-m", "SEGMENT", "-st", "topological", "-sr", "n"], "shapegraph-map-csv", 0.0, 0.0),
    "seg_metric": (["-m", "SEGMENT", "-st", "metric", "-sr", "n"], "shapegraph-map-csv", 0.0, 0.0),
    "seg_topological_r": (["-m", "SEGMENT", "-st", "topological", "-sr", "400"], "shapegraph-map-csv", 0.0, 0.0),
    "seg_metric_r": (["-m", "SEGMENT", "-st", "metric", "-sr", "400"], "shapegraph-map-csv", 0.0, 0.0),
    "axial": (["-m", "AXIAL", "-xa", "n", "-xac", "-xal", "-xar"], "shapegraph-map-csv", 0.0, 0.0),
    "axial_r": (["-m", "AXIAL", "-xa", "3,n", "-xac", "-xar"], "shapegraph-map-csv", 0.0, 0.0),
    "axial_plain": (["-m", "AXIAL", "-xa", "n", "-xal", "-xar"], "shapegraph-map-csv", 0.0, 0.0),
    "axial_w": (["-m", "AXIAL", "-xa", "3,n", "-xac", "-xar", "-xaw", "Connectivity"], "shapegraph-map-csv", 0.0, 0.0),
    "axial_brandes": (["-m", "AXIAL", "-xa", "3,n", "-xac", "-xacb"], "shapegraph-map-csv", 0.0, 0.0),
    "vis_global_list": (["-m", "VGA", "-vm", "visibility", "-vg", "-vl", "-vr", "3,5,n"], "pointmap-data-csv", 0.0,
                        0.0),
    "vis_global_list_rev": (["-m", "VGA", "-vm", "visibility", "-vg", "-vr", "n,5,3"], "pointmap-data-csv", 0.0, 0.0),
    "vis_both": (["-m", "VGA", "-vm", "visibility", "-vg", "-vl", "-vr", "n"], "pointmap-data-csv", 0.0, 0.0),
}

MODE_FLAGS = {"serial": ["-serial"], "cpu": [], "gpu": ["-gpu", "-gpumin", "0"], "rapido": []}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run(cmd, log_path):
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def analysis_seconds(times_csv):
    with open(times_csv, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[0].strip('"') not in ("action", "Load graph file", "Writing graph"):
                return float(parts[1])
    return float("nan")


def make_grid(cli, base_graph, spacing, fill, out_dir):
    path = os.path.join(out_dir, f"grid_{spacing}.graph")
    if not os.path.exists(path):
        run([cli, "-m", "VISPREP", "-f", base_graph, "-o", path, "-pg", spacing, "-pp", fill, "-pm"],
            os.path.join(out_dir, f"grid_{spacing}.log"))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", required=True, help="new depthmapXcli.exe")
    ap.add_argument("--rapido", help="rapido depthmapXcli.exe (enables the 'rapido' mode)")
    ap.add_argument("--graph", action="append", default=[], help="input .graph with a VGA grid (repeatable)")
    ap.add_argument("--make-grid", action="append", default=[], metavar="SPACING",
                    help="also generate a grid with this spacing from --grid-base (repeatable)")
    ap.add_argument("--grid-base", help="graph with drawing layers used by --make-grid")
    ap.add_argument("--grid-fill", default="0.68,7.04", help="fill point for --make-grid")
    ap.add_argument("--cases", default="vis_local", help="comma-separated: " + ",".join(CASES))
    ap.add_argument("--modes", default="serial,cpu,gpu")
    ap.add_argument("--reference", default="serial", help="mode the others are compared with")
    ap.add_argument("--out", required=True, help="working directory for outputs")
    ap.add_argument("--csv-only", action="store_true",
                    help="compare exported attributes only, not the .graph file bytes (e.g. against the "
                         "rapido build, whose .graph may store columns differently)")
    ap.add_argument("--gpu-optional", action="store_true",
                    help="do not fail when the gpu mode falls back to the CPU (e.g. maps with merge links)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    graphs = list(a.graph)
    for spacing in a.make_grid:
        if not a.grid_base:
            ap.error("--make-grid needs --grid-base")
        graphs.append(make_grid(a.cli, a.grid_base, spacing, a.grid_fill, a.out))

    modes = a.modes.split(",")
    if "rapido" in modes and not a.rapido:
        ap.error("mode 'rapido' needs --rapido")
    if a.reference not in modes:
        ap.error("--reference must be one of --modes")
    ordered = [a.reference] + [m for m in modes if m != a.reference]

    rows = []
    all_ok = True
    for graph in graphs:
        gname = os.path.splitext(os.path.basename(graph))[0]
        for case in a.cases.split(","):
            args, export_mode, rtol, atol = CASES[case]
            ref_csv = None
            ref_graph = None
            for mode in ordered:
                exe = a.rapido if mode == "rapido" else a.cli
                stem = os.path.join(a.out, f"{gname}.{case}.{mode}")
                out = run([exe, *args, *MODE_FLAGS[mode], "-f", graph, "-o", stem + ".graph", "-t", stem + ".times.csv"],
                          stem + ".log")
                secs = analysis_seconds(stem + ".times.csv")
                run([a.cli, "-m", "EXPORT", "-f", stem + ".graph", "-o", stem + ".csv", "-em", export_mode],
                    stem + ".export.log")
                note = ""
                if mode == "gpu" and "GPU (" not in out:
                    note = "GPU NOT USED"
                    all_ok = all_ok and a.gpu_optional
                if ref_csv is None:
                    ref_csv, verdict = stem + ".csv", "reference"
                    ref_graph = sha256(stem + ".graph")
                else:
                    with open(stem + ".compare.txt", "w", encoding="utf-8") as f:
                        old = sys.stdout
                        sys.stdout = f
                        try:
                            sys.argv = ["compare_outputs", ref_csv, stem + ".csv", "--rtol", str(rtol), "--atol",
                                        str(atol)]
                            rc = compare_outputs.main()
                        finally:
                            sys.stdout = old
                    verdict = "MATCH" if rc == 0 else "DIFFERENT (see .compare.txt)"
                    all_ok = all_ok and rc == 0
                    if not a.csv_only:
                        same_graph = sha256(stem + ".graph") == ref_graph
                        verdict += ", .graph identical" if same_graph else ", .graph BYTES DIFFER"
                        all_ok = all_ok and same_graph
                reports = [line for line in out.splitlines() if line.startswith(("VGA ", "Segment ", "Axial "))]
                rows.append((gname, case, mode, secs, verdict, note or "; ".join(reports)))
                print(f"{gname:22} {case:13} {mode:7} {secs:10.3f} s  {verdict}  {note}", flush=True)

    print()
    print(f"| graph | case | mode | analysis time (s) | vs {a.reference} | path |")
    print("|---|---|---|---:|---|---|")
    for g, c, m, s, v, n in rows:
        print(f"| {g} | {c} | {m} | {s:.3f} | {v} | {n} |")
    print("\nALL MATCH" if all_ok else "\nVALIDATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
