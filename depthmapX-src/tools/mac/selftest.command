#!/bin/sh
# depthmapX multicore + GPU for macOS: self-test. Double-click it in Finder.
#
# Runs each analysis with the original single-core code (-serial), on all cores, on the
# GPU (Metal, VGA visibility only) and with upstream's official depthmapXcli 0.9.1, then
# compares the .graph files and the exported attribute tables byte for byte. Everything
# is written to test-report.txt next to this file (send that file back), the outputs to
# test-results/. Only the system's sh, perl and cmp are used.

DIR="$(cd "$(dirname "$0")" && pwd)"
CLI="$DIR/motor/depthmapXcli"
EX="$DIR/exemplos"
OUT="$DIR/test-results"
REPORT="$DIR/test-report.txt"
FAILS="$OUT/.fails"

now() { perl -MTime::HiRes=time -e 'printf "%.2f", time'; }
elapsed() { perl -e "printf '%7.2f', $2 - $1"; }

# run <label> <program> <args...>: runs it, keeps its log, prints time and the path line.
run() {
    rl=$1
    shift
    t0=$(now)
    "$@" > "$OUT/$rl.log" 2>&1
    rc=$?
    t1=$(now)
    paths=$(grep -E ': (CPU|GPU)' "$OUT/$rl.log" | sed 's/^/        /')
    if [ $rc -ne 0 ]; then
        echo "  $(printf '%-34s' "$rl") FAILED TO RUN (exit code $rc), see test-results/$rl.log"
        tail -5 "$OUT/$rl.log" | sed 's/^/        /'
        echo "$rl" >> "$FAILS"
        return 1
    fi
    echo "  $(printf '%-34s' "$rl") $(elapsed "$t0" "$t1") s"
    [ -n "$paths" ] && echo "$paths"
    return 0
}

# same <label> <reference label> <kind>: compares the .graph and .csv of two runs.
# kind "must" -> a difference is a FAIL; kind "info" -> reported only.
same() {
    if [ ! -f "$OUT/$1.graph" ] || [ ! -f "$OUT/$2.graph" ]; then
        return
    fi
    gs=identical
    cmp -s "$OUT/$1.graph" "$OUT/$2.graph" || gs=DIFFERENT
    cs=identical
    cmp -s "$OUT/$1.csv" "$OUT/$2.csv" || cs=DIFFERENT
    verdict=ok
    if [ "$gs$cs" != identicalidentical ]; then
        if [ "$3" = must ]; then
            verdict=FAIL
            echo "$1 vs $2" >> "$FAILS"
        else
            verdict="differs (see note)"
        fi
    fi
    echo "  $(printf '%-34s' "$1 = $2"): .graph $gs, table $cs -> $verdict"
}

# analysis <name> <input> <export mode> <modes> <args...>
# modes: letters s (serial), m (multicore), g (GPU), u (upstream official)
analysis() {
    name=$1 input=$2 export=$3 modes=$4
    shift 4
    echo
    echo "== $name: $*"
    for mode in s m g u; do
        case "$modes" in *$mode*) ;; *) continue ;; esac
        case $mode in
            s) label="$name-serial";    prog="$CLI"; extra="-serial" ;;
            m) label="$name-multicore"; prog="$CLI"; extra="" ;;
            g) label="$name-gpu";       prog="$CLI"; extra="-gpu -gpumin 0" ;;
            u) label="$name-upstream";  prog="$UPSTREAM"; extra="" ;;
        esac
        [ $mode = u ] && [ ! -x "$UPSTREAM" ] && continue
        label="$label$SUFFIX"
        # $PREFIX and $extra are split into words on purpose
        if run "$label" $PREFIX "$prog" -f "$EX/$input" -o "$OUT/$label.graph" "$@" $extra; then
            run "$label-export" $PREFIX "$CLI" -f "$OUT/$label.graph" -o "$OUT/$label.csv" -m EXPORT -em "$export" > /dev/null
        fi
    done
    case "$modes" in *m*) same "$name-multicore$SUFFIX" "$name-serial$SUFFIX" must ;; esac
    case "$modes" in *g*) same "$name-gpu$SUFFIX" "$name-serial$SUFFIX" must ;; esac
    case "$modes" in *u*) same "$name-serial$SUFFIX" "$name-upstream$SUFFIX" info ;; esac
}

suite() {
    analysis vga-visibility    galeria_grade_0.04.graph pointmap-data-csv  smgu -m VGA -vm visibility -vl -vg -vr n
    analysis vga-visibility-r3 galeria_grade_0.04.graph pointmap-data-csv  smgu -m VGA -vm visibility -vg -vr 3
    analysis vga-radius-list   galeria_grade_0.04.graph pointmap-data-csv  smg  -m VGA -vm visibility -vg -vr 3,5,n
    analysis vga-metric        galeria_grade_0.04.graph pointmap-data-csv  smu  -m VGA -vm metric -vr n
    analysis vga-angular       galeria_grade_0.04.graph pointmap-data-csv  smu  -m VGA -vm angular
    analysis visprep           galeria_vazia.graph      pointmap-data-csv  smu  -m VISPREP -pg 0.04 -pp 0.68,7.04 -pm
    analysis axial             barnsbury_axial.graph    shapegraph-map-csv smu  -m AXIAL -xa 3,n -xac -xal -xar
    analysis axial-brandes     barnsbury_axial.graph    shapegraph-map-csv sm   -m AXIAL -xa 3,n -xac -xacb
    analysis segment-tulip     barnsbury_segmentos.graph shapegraph-map-csv smu -m SEGMENT -st tulip -sr n -srt metric -sic -stb 1024
    analysis segment-angular   barnsbury_segmentos.graph shapegraph-map-csv smu -m SEGMENT -st angular -sr n
    analysis segment-metric    barnsbury_segmentos.graph shapegraph-map-csv smu -m SEGMENT -st metric -sr n
}

main() {
    echo "depthmapX multicore + GPU for macOS - self-test, $(date)"
    echo
    echo "== This Mac"
    sw_vers | sed 's/^/  /'
    echo "  architecture: $(uname -m); CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null); cores: $(sysctl -n hw.ncpu)"
    mem=$(sysctl -n hw.memsize 2>/dev/null)
    [ -n "$mem" ] && echo "  memory: $(perl -e "printf '%.0f', $mem / 1073741824") GiB"
    system_profiler SPDisplaysDataType 2>/dev/null | grep -E 'Chipset Model|Metal|Total Number of Cores' | sed 's/^ */  /'
    echo
    echo "== Programs"
    for f in "$CLI" "$DIR/motor/libomp.dylib" "$DIR/motor/libdepthmapX_gpu.dylib"; do
        echo "  $(basename "$f"): $(file -b "$f" 2>/dev/null | cut -c1-90)"
    done
    if xattr "$CLI" 2>/dev/null | grep -q quarantine; then
        echo "  WARNING: the files are still quarantined; if anything fails to start, see LEIA-ME README.txt"
    fi
    "$CLI" -h > "$OUT/help.log" 2>&1 || { echo "  depthmapXcli does not start:"; sed 's/^/    /' "$OUT/help.log"; }

    case "$(uname -m)" in
        arm64) NATIVE=arm64; UPSTREAM_NATIVE="$DIR/teste/upstream-0.9.1/depthmapXcli-arm64" ;;
        *)     NATIVE=x86_64; UPSTREAM_NATIVE="$DIR/teste/upstream-0.9.1/depthmapXcli-intel" ;;
    esac

    echo
    echo "######## $NATIVE (native)"
    PREFIX="" SUFFIX="" UPSTREAM="$UPSTREAM_NATIVE"
    suite

    echo
    echo "== GPU speed (VGA visibility, local + global), larger grids: all cores vs GPU"
    for spacing in 0.02 0.015; do
        g="grid-$spacing"
        run "$g-visprep" "$CLI" -f "$EX/galeria_vazia.graph" -o "$OUT/$g.graph" -m VISPREP -pg $spacing -pp 0.68,7.04 -pm > /dev/null
        for mode in multicore gpu; do
            extra=""
            [ $mode = gpu ] && extra="-gpu -gpumin 0"
            if run "$g-$mode" "$CLI" -f "$OUT/$g.graph" -o "$OUT/$g-$mode.graph" -m VGA -vm visibility -vl -vg -vr n $extra; then
                run "$g-$mode-export" "$CLI" -f "$OUT/$g-$mode.graph" -o "$OUT/$g-$mode.csv" -m EXPORT -em pointmap-data-csv > /dev/null
            fi
        done
        same "$g-gpu" "$g-multicore" must
    done

    if [ $NATIVE = arm64 ] && arch -x86_64 /usr/bin/true 2>/dev/null; then
        echo
        echo "######## x86_64 (Intel code under Rosetta 2)"
        PREFIX="arch -x86_64" SUFFIX="-x86_64" UPSTREAM="$DIR/teste/upstream-0.9.1/depthmapXcli-intel"
        analysis vga-visibility galeria_grade_0.04.graph pointmap-data-csv  smgu -m VGA -vm visibility -vl -vg -vr n
        analysis vga-metric     galeria_grade_0.04.graph pointmap-data-csv  smu  -m VGA -vm metric -vr n
        analysis axial          barnsbury_axial.graph    shapegraph-map-csv smu  -m AXIAL -xa 3,n -xac -xal -xar
        analysis segment-tulip  barnsbury_segmentos.graph shapegraph-map-csv smu -m SEGMENT -st tulip -sr n -srt metric -sic -stb 1024
        echo
        echo "== arm64 = x86_64 (same results on both processors)"
        for n in vga-visibility-serial vga-visibility-multicore vga-visibility-gpu vga-metric-multicore axial-multicore segment-tulip-multicore; do
            same "$n-x86_64" "$n" must
        done
    elif [ $NATIVE = arm64 ]; then
        echo
        echo "(Rosetta 2 is not installed: the Intel version was not tested on this Mac)"
    fi

    echo
    echo "######## Summary"
    if [ -s "$FAILS" ]; then
        echo "  FAILED: $(wc -l < "$FAILS" | tr -d ' ') check(s):"
        sed 's/^/    /' "$FAILS"
    else
        echo "  PASSED: every multicore and GPU result is identical to the original single-core code."
    fi
    echo "  Note: \"differs\" against upstream's official program is reported for information only;"
    echo "  its own compiler settings may round some decimals differently."
    echo
    echo "  Please send this report: $REPORT"
}

rm -rf "$OUT"
mkdir -p "$OUT"
: > "$FAILS"
main 2>&1 | tee "$REPORT"
