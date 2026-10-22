#!/bin/bash
# Compiles the Metal kernels with Apple's offline Metal compiler (comes with Xcode), so
# Metal-only errors show up even on a Mac without a usable GPU. The module compiles the
# same source at run time (dmx_metal_shared.h + dmx_metal_kernels.metal, joined as by
# gpu/CMakeLists.txt); this checks it ahead of time:
#  1. with the SDK's default Metal language version (must succeed);
#  2. as Metal 2.3 for macOS 11, the oldest system we support (reported; a failure means
#     the kernels need a newer macOS than the README says).
#
#   tools/mac/check_metal_kernels.sh [output dir]
set -uo pipefail

SRC="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-${TMPDIR:-/tmp}/dmx-metal-check}"
mkdir -p "$OUT"
cat "$SRC/gpu/metal/dmx_metal_shared.h" > "$OUT/kernels.metal"
printf '\n' >> "$OUT/kernels.metal"
cat "$SRC/gpu/metal/dmx_metal_kernels.metal" >> "$OUT/kernels.metal"

metal() { xcrun -sdk macosx metal "$@"; }

# Xcode 26 and later install the Metal toolchain on demand.
if ! metal --version > /dev/null 2>&1; then
    echo "Metal compiler not installed; downloading the Metal toolchain"
    xcodebuild -downloadComponent MetalToolchain || true
fi
metal --version || { echo "FAILED: no Metal compiler (xcrun metal)"; exit 1; }

echo "### default language version"
if ! metal -c "$OUT/kernels.metal" -o "$OUT/kernels.air" -Wall; then
    echo "FAILED: the Metal kernels do not compile"
    exit 1
fi
echo "compiled: $OUT/kernels.air"
for k in dmx_probe dmx_vis_local_rows dmx_vis_global_pull dmx_vis_global_downsweep dmx_vis_global_combine \
         dmx_vis_global_count; do
    # names are normally stored as plain text in the bitcode; a miss is worth a look
    grep -q "$k" "$OUT/kernels.air" || echo "WARNING: kernel name $k not found in the compiled file"
done

echo "### Metal 2.3 (macOS 11)"
if metal -std=macos-metal2.3 -mmacosx-version-min=11.0 -c "$OUT/kernels.metal" -o "$OUT/kernels-2.3.air"; then
    echo "compiled as Metal 2.3"
else
    echo "WARNING: not accepted as Metal 2.3 for macOS 11 (see the messages above)"
fi
echo "PASSED"
