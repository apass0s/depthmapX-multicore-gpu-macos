#!/bin/bash
# Tests a native macOS build (tools/mac/build_macos.sh) on the Mac it runs on:
#  1. the unit test suites;
#  2. builds UNMODIFIED upstream 0.9.1 with the same compiler, flags and architecture
#     (tools/mac/upstream-cli) and compares every analysis with it, bit for bit:
#     exported attributes and .graph bytes, for -serial, multicore and GPU (Metal) runs;
#  3. the Metal module on larger grids against the multicore CPU.
#
#   tools/mac/test_macos.sh <build dir> <upstream depthmapX 0.9.1 source> <examples dir> [arch]
#
# <examples dir>: the exemplos folder of the Windows release (galeria_grade_0.04.graph,
# galeria_vazia.graph, barnsbury_axial.graph, barnsbury_segmentos.graph). Upstream source:
# https://github.com/SpaceGroupUCL/depthmapX/archive/refs/tags/v0.9.1.tar.gz
# Writes everything to <build dir>/macos-tests and ends with PASSED or FAILED.
set -uo pipefail

SRC="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$(cd "$1" && pwd)"
UPSTREAM="$(cd "$2" && pwd)"
EX="$(cd "$3" && pwd)"
ARCH="${4:-$(uname -m)}"
OUT="$BUILD/macos-tests"
TD="$SRC/testdata"
mkdir -p "$OUT"
FAILED=0
fail() { echo "FAILED: $*"; FAILED=1; }

echo "### 1. unit tests"
for t in genlibTest salaTest cliTest moduleTest mgraph440Test depthmapXTest GuiUnitTest; do
    exe="$BUILD/$t/$t"
    [ -x "$exe" ] || { fail "$t not built"; continue; }
    if (cd "$BUILD/$t" && QT_QPA_PLATFORM=offscreen "./$t" > "$OUT/$t.log" 2>&1); then
        echo "$t: $(tail -n 2 "$OUT/$t.log" | tr '\n' ' ')"
    else
        fail "$t (see $OUT/$t.log)"
    fi
done

echo "### 2. upstream reference (same compiler and flags)"
UPBUILD="$OUT/upstream-build"
CMAKE="$(command -v cmake)"
if ! "$CMAKE" -S "$SRC/tools/mac/upstream-cli" -B "$UPBUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES="$ARCH" -DUPSTREAM_DIR="$UPSTREAM" > "$OUT/upstream-build.log" 2>&1 ||
   ! "$CMAKE" --build "$UPBUILD" --target depthmapXcli >> "$OUT/upstream-build.log" 2>&1; then
    fail "upstream build (see $OUT/upstream-build.log)"
fi
CLI="$BUILD/depthmapXcli/depthmapXcli"
UP="$UPBUILD/depthmapXcli/depthmapXcli"
"$CLI" -m VGA -f "$EX/galeria_grade_0.04.graph" -o "$OUT/probe.graph" -vm visibility -vl -gpu -gpumin 0 \
    | grep "VGA visual local" || true

V() { python3 "$SRC/tools/gpu/validate.py" --cli "$CLI" --upstream "$UP" --out "$OUT/validate" "$@" \
          > "$OUT/validate-$CASE.log" 2>&1 || fail "$CASE (see $OUT/validate-$CASE.log)";
      grep -v '^|' "$OUT/validate-$CASE.log" | grep . ; }
CASE=vis;        V --graph "$EX/galeria_grade_0.04.graph" --modes upstream,serial,cpu,gpu --reference upstream \
                   --cases vis_local,vis_global,vis_global_r3,vis_both
CASE=vis_lists;  V --graph "$EX/galeria_grade_0.04.graph" --modes serial,cpu,gpu --reference serial \
                   --cases vis_global_list,vis_global_list_rev
CASE=vis_links;  V --graph "$TD/turns_connected.graph" --graph "$TD/gallery_connected.graph" \
                   --modes upstream,serial,cpu,gpu --reference upstream --gpu-optional --cases vis_local,vis_global,vis_both
CASE=vga_paths;  V --graph "$EX/galeria_grade_0.04.graph" --graph "$TD/turns_connected.graph" \
                   --modes upstream,serial,cpu --reference upstream --cases metric,metric_r,angular
CASE=axial;      V --graph "$EX/barnsbury_axial.graph" --graph "$TD/barnsbury_extended1_axial.graph" \
                   --modes upstream,cpu --reference upstream --cases axial,axial_r,axial_plain,axial_w
CASE=brandes;    V --graph "$EX/barnsbury_axial.graph" --modes serial,cpu --reference serial --cases axial_brandes
CASE=segment;    V --graph "$EX/barnsbury_segmentos.graph" --modes upstream,cpu --reference upstream \
                   --cases seg_angular,seg_angular_r,seg_tulip,seg_tulip_r,seg_tulip_w,seg_tulip_steps,seg_tulip_ang,seg_topological,seg_metric,seg_topological_r,seg_metric_r
echo "### VISPREP (visibility graph construction)"
for sp in 0.04 0.02; do
    "$UP" -m VISPREP -f "$EX/galeria_vazia.graph" -o "$OUT/visprep_${sp}_upstream.graph" -pg $sp -pp 0.68,7.04 -pm > /dev/null
    "$CLI" -m VISPREP -f "$EX/galeria_vazia.graph" -o "$OUT/visprep_${sp}_ours.graph" -pg $sp -pp 0.68,7.04 -pm > /dev/null
    if [ -s "$OUT/visprep_${sp}_ours.graph" ] && cmp -s "$OUT/visprep_${sp}_upstream.graph" "$OUT/visprep_${sp}_ours.graph"; then
        echo "visprep $sp: .graph identical"
    else
        fail "visprep $sp: .graph bytes differ"
    fi
done

echo "### 3. Metal on larger grids (against the multicore CPU)"
CASE=metal_large; V --make-grid 0.02 --make-grid 0.01 --grid-base "$EX/galeria_vazia.graph" --modes cpu,gpu \
                    --reference cpu --cases vis_local,vis_global,vis_global_r3,vis_global_list

echo
[ $FAILED = 0 ] && echo "PASSED ($ARCH)" || echo "FAILED ($ARCH): see the lines above"
exit $FAILED
