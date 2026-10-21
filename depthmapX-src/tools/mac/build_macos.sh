#!/bin/bash
# Native macOS build of depthmapX (GUI + command line + Metal module + tests) for one
# architecture. Run on a Mac, from anywhere:
#
#   tools/mac/build_macos.sh [arm64|x86_64] [build dir]
#
# Defaults: this Mac's architecture, build-macos-<arch> next to the source folder.
# Needs the Xcode Command Line Tools (xcode-select --install) and, from Homebrew,
# cmake, ninja, qt@5 and libomp for the target architecture:
#   Apple Silicon, arm64 build:  /opt/homebrew/bin/brew install cmake ninja qt@5 libomp
#   Intel build (on an Intel Mac, or on Apple Silicon under Rosetta with the Intel
#   Homebrew in /usr/local):     arch -x86_64 /usr/local/bin/brew install cmake ninja qt@5 libomp
# Other locations: set QT_PREFIX and LIBOMP_PREFIX.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/../.." && pwd)"
ARCH="${1:-$(uname -m)}"
case "$ARCH" in
    arm64) BREW=/opt/homebrew ;;
    x86_64) BREW=/usr/local ;;
    *) echo "architecture must be arm64 or x86_64, not $ARCH" >&2; exit 1 ;;
esac
BUILD="${2:-$SRC/../build-macos-$ARCH}"
QT_PREFIX="${QT_PREFIX:-$BREW/opt/qt@5}"
LIBOMP_PREFIX="${LIBOMP_PREFIX:-$BREW/opt/libomp}"

for f in "$QT_PREFIX/lib/cmake/Qt5/Qt5Config.cmake" "$LIBOMP_PREFIX/include/omp.h"; do
    [ -e "$f" ] || { echo "missing $f (see the comments at the top of $0)" >&2; exit 1; }
done
for lib in "$QT_PREFIX/lib/QtCore.framework/QtCore" "$LIBOMP_PREFIX/lib/libomp.dylib"; do
    if ! lipo -archs "$lib" | tr ' ' '\n' | grep -qx "$ARCH"; then
        echo "$lib has no $ARCH code ($(lipo -archs "$lib")): install the $ARCH Homebrew packages" >&2
        exit 1
    fi
done

# An x86_64 build on Apple Silicon needs Rosetta 2 (softwareupdate --install-rosetta):
# macOS runs the x86_64 Qt tools and the x86_64 tests under it automatically.
CMAKE="$(command -v cmake)"

"$CMAKE" -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES="$ARCH" -DCMAKE_PREFIX_PATH="$QT_PREFIX" \
    -DDEPTHMAPX_LIBOMP_ROOT="$LIBOMP_PREFIX"
"$CMAKE" --build "$BUILD"

echo
echo "Built for $ARCH in $BUILD:"
file "$BUILD/depthmapXcli/depthmapXcli" "$BUILD/depthmapXcli/libdepthmapX_gpu.dylib" \
     "$BUILD/depthmapX/depthmapX.app/Contents/MacOS/depthmapX"
