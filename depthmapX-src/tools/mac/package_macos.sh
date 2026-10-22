#!/bin/bash
# Packages a native macOS build (tools/mac/build_macos.sh) as a folder and a zip:
#
#   depthmapX-multicore-gpu-macos-<arch>/
#     interface/depthmapX.app   GUI, with Qt, libomp and the Metal module inside
#     motor/                    depthmapXcli + libdepthmapX_gpu.dylib + libomp.dylib
#     exemplos/, LEIA-ME README.txt, Terminal depthmapX.command
#
#   tools/mac/package_macos.sh <build dir> <examples dir> [arch] [output dir]
#
# Run on the Mac that built it. Binaries are signed ad hoc (required on Apple Silicon
# after macdeployqt and install_name_tool change them); not notarised, so users clear
# the quarantine flag once (see the README).
set -euo pipefail

SRC="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$(cd "$1" && pwd)"
EX="$(cd "$2" && pwd)"
ARCH="${3:-$(uname -m)}"
OUTDIR="${4:-$SRC/..}"
case "$ARCH" in
    arm64) BREW=/opt/homebrew ;;
    x86_64) BREW=/usr/local ;;
    *) echo "architecture must be arm64 or x86_64" >&2; exit 1 ;;
esac
QT_PREFIX="${QT_PREFIX:-$BREW/opt/qt@5}"
LIBOMP_PREFIX="${LIBOMP_PREFIX:-$BREW/opt/libomp}"

NAME="depthmapX-multicore-gpu-macos-$ARCH"
DIST="$OUTDIR/$NAME"
[ -e "$DIST" ] && { echo "$DIST exists: remove it first" >&2; exit 1; }
mkdir -p "$DIST/interface" "$DIST/motor" "$DIST/exemplos"

echo "## interface"
cp -R "$BUILD/depthmapX/depthmapX.app" "$DIST/interface/"
APP="$DIST/interface/depthmapX.app"
[ -f "$APP/Contents/Frameworks/libdepthmapX_gpu.dylib" ] || { echo "Metal module missing from the app" >&2; exit 1; }
# Qt frameworks and plugins, and every non-system library (libomp) into the bundle.
"$QT_PREFIX/bin/macdeployqt" "$APP" -always-overwrite
codesign --force --deep --sign - "$APP"

echo "## motor"
cp "$BUILD/depthmapXcli/depthmapXcli" "$BUILD/depthmapXcli/libdepthmapX_gpu.dylib" "$DIST/motor/"
cp -L "$LIBOMP_PREFIX/lib/libomp.dylib" "$DIST/motor/libomp.dylib"
chmod u+w "$DIST/motor/"*
# depthmapXcli was linked against Homebrew's libomp by its absolute install name; make it
# load the copy next to it (its rpath is @executable_path, see CMakeLists.txt).
OLD_ID="$(otool -D "$LIBOMP_PREFIX/lib/libomp.dylib" | tail -n 1)"
install_name_tool -change "$OLD_ID" "@rpath/libomp.dylib" "$DIST/motor/depthmapXcli"
install_name_tool -id "@rpath/libomp.dylib" "$DIST/motor/libomp.dylib"
for f in "$DIST/motor/"*; do
    codesign --force --sign - "$f"
done

echo "## documents and examples"
find "$EX" -maxdepth 1 -type f \( -name '*.graph' -o -name '*.dxf' \) -exec cp {} "$DIST/exemplos/" \;
STATUS="Built on macOS ($(sw_vers -productVersion), $ARCH) from the macos-testing branch; validated
with tools/mac/test_macos.sh on that Mac (see its report)."
FOLDERS="  interface/depthmapX.app    graphical interface, with the multicore and GPU code
  motor/                     depthmapXcli (command line) and the Metal GPU module
  Teste depthmapX.command    double-click: self-test, writes test-report.txt"
python3 - "$SRC/tools/mac/README-macos.txt" "$DIST/LEIA-ME README.txt" "$STATUS" "$FOLDERS" <<'PY'
import sys
src, dst, status, folders = sys.argv[1:]
text = open(src, encoding="utf-8").read().replace("{STATUS}", status).replace("{FOLDERS}", folders)
open(dst, "w", encoding="utf-8").write(text)
PY
cat > "$DIST/Terminal depthmapX.command" <<'SH'
#!/bin/sh
DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$DIR/motor:$PATH"
cd "$DIR/exemplos"
echo
echo " depthmapX multicore + GPU - depthmapXcli ready. See LEIA-ME README.txt"
echo " Help: depthmapXcli -h"
echo
exec "${SHELL:-/bin/zsh}" -i
SH
chmod +x "$DIST/Terminal depthmapX.command"
cp "$SRC/tools/mac/selftest.command" "$DIST/Teste depthmapX.command"
chmod +x "$DIST/Teste depthmapX.command"
# Upstream's official depthmapXcli 0.9.1 (release assets depthmapXcli_macos_arm64 and
# depthmapXcli_macos_intel), the self-test's independent reference, when given.
if [ -n "${UPSTREAM_CLI_DIR:-}" ]; then
    mkdir -p "$DIST/teste/upstream-0.9.1"
    cp "$UPSTREAM_CLI_DIR/depthmapXcli_macos_arm64" "$DIST/teste/upstream-0.9.1/depthmapXcli-arm64"
    cp "$UPSTREAM_CLI_DIR/depthmapXcli_macos_intel" "$DIST/teste/upstream-0.9.1/depthmapXcli-intel"
    chmod +x "$DIST/teste/upstream-0.9.1/"*
fi

echo "## checks"
otool -L "$DIST/motor/depthmapXcli"
"$DIST/motor/depthmapXcli" -m VGA -f "$DIST/exemplos/galeria_grade_0.04.graph" -o /tmp/dmx-package-check.graph \
    -vm visibility -vl -vg -vr n -gpu -gpumin 0 | grep "^VGA "
codesign --verify --deep --strict "$APP" && echo "app signature ok"

echo "## zip"
(cd "$OUTDIR" && ditto -c -k --sequesterRsrc --keepParent "$NAME" "$NAME.zip")
echo "Done: $DIST and $DIST.zip"
