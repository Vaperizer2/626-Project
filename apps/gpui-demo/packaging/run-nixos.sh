#!/usr/bin/env bash
# Component Classifier: launcher for NixOS (and any system without an FHS /lib64), e.g. niri / sway / Hyprland.
#
# Why this exists: the bundled binary is a normal "generic Linux" executable. On NixOS
#   * there is no /lib64/ld-linux-x86-64.so.2 and no /usr/lib, so it cannot start at all (unless nix-ld is set up);
#   * GPUI loads libwayland-client / libEGL / libvulkan at runtime with dlopen; when they are not found it silently
#     falls back to X11, and niri has no built-in X server (only xwayland-satellite), so the window never appears.
# This script takes the needed libraries from nixpkgs (no root, nothing installed system-wide), adds the system GPU
# drivers from /run/opengl-driver, and starts the app natively on Wayland.
#
# Usage (from anywhere; the folder can live anywhere):
#   ./run-nixos.sh                       open the window (Wayland)
#   ./run-nixos.sh --tab demo --dark     any app option is passed through (see ./run-nixos.sh --help)
#   ./run-nixos.sh classify demo_stack/01_battery_holder.jpg   headless check, no display needed
#   ./run-nixos.sh --x11 ...             use X11 instead (needs a running X server, e.g. xwayland-satellite on niri)
#   ./run-nixos.sh --doctor              print what was found and exit
# Environment overrides:
#   NIXPKGS=<flake-ref>   nixpkgs to take libraries from (default: your system's `nixpkgs`, i.e. the channel <nixpkgs>
#                         or the flake registry entry; using the SAME nixpkgs as your system keeps the GPU drivers'
#                         glibc compatible). Example: NIXPKGS=github:NixOS/nixpkgs/nixos-25.05
set -euo pipefail

HERE=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BIN="$HERE/gpui-demo"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/component-classifier"
mkdir -p "$CACHE"

say() { printf 'run-nixos.sh: %s\n' "$*" >&2; }
die() { say "$*"; exit 1; }

USE_X11=0
DOCTOR=0
ARGS=()
for a in "$@"; do
    case "$a" in
        --x11) USE_X11=1 ;;
        --doctor) DOCTOR=1 ;;
        *) ARGS+=("$a") ;;
    esac
done

[ "$(uname -m)" = "x86_64" ] || die "this bundle is built for x86_64 Linux; this machine is $(uname -m)."
[ -f "$BIN" ] || die "$BIN is missing (was the archive fully extracted?)."
command -v nix >/dev/null 2>&1 || die "nix is not on PATH. This launcher is for NixOS; on other distros use ./run.sh."

# ---------------------------------------------------------------- resolve libraries from nixpkgs (cached)
# attribute -> what we need from it
#   glibc                 ld-linux-x86-64.so.2 (the program interpreter) + libc/libm/libdl/libpthread/librt
#   stdenv.cc.cc.lib      libstdc++.so.6, libgcc_s.so.1 (onnxruntime)
#   wayland               libwayland-client.so.0, libwayland-egl.so.1, libwayland-cursor
#   libxkbcommon          libxkbcommon.so.0 (+ -x11), keyboard layouts compiled in
#   libxcb                libxcb.so.1 (linked even on Wayland)
#   libglvnd              libEGL.so.1 (dispatches to the system Mesa/NVIDIA driver in /run/opengl-driver)
#   vulkan-loader         libvulkan.so.1 (finds the system ICDs in /run/opengl-driver)
ATTRS=(glibc stdenv.cc.cc.lib wayland libxkbcommon libxcb libglvnd vulkan-loader)

nix_cmd() { nix --extra-experimental-features 'nix-command flakes' "$@"; }

resolve_one() { # $1 = attribute; prints the store path of its library output
    local attr="$1" alt="" p=""
    [ "$attr" = libxcb ] && alt="xorg.libxcb"
    for a in "$attr" $alt; do
        if [ -n "${NIXPKGS:-}" ]; then
            p=$(nix_cmd build --no-link --print-out-paths "${NIXPKGS}#${a}" 2>/dev/null | head -n1) && [ -n "$p" ] && break
        else
            if nix-instantiate --find-file nixpkgs >/dev/null 2>&1; then
                p=$(nix-build '<nixpkgs>' -A "$a" --no-out-link 2>/dev/null | head -n1) && [ -n "$p" ] && break
            fi
            p=$(nix_cmd build --no-link --print-out-paths "nixpkgs#${a}" 2>/dev/null | head -n1) && [ -n "$p" ] && break
        fi
        p=""
    done
    [ -n "$p" ] || return 1
    printf '%s\n' "$p"
}

KEY="${NIXPKGS:-system}"; KEY="${KEY//[^A-Za-z0-9._-]/_}"
LIBFILE="$CACHE/libpaths-$KEY.txt"
paths_ok() { [ -s "$LIBFILE" ] && while read -r d; do [ -d "$d" ] || return 1; done < "$LIBFILE"; }
if ! paths_ok; then
    say "first run: fetching runtime libraries from nixpkgs (one-off, cached in $CACHE) ..."
    : > "$LIBFILE.tmp"
    for attr in "${ATTRS[@]}"; do
        p=$(resolve_one "$attr") || {
            rm -f "$LIBFILE.tmp"
            die "could not get '$attr' from nixpkgs. Is a nixpkgs channel or the flake registry available?
  Try:  NIXPKGS=github:NixOS/nixpkgs/nixos-unstable $0 ${ARGS[*]:-}"
        }
        [ -d "$p/lib" ] || { rm -f "$LIBFILE.tmp"; die "'$attr' -> $p has no lib/ directory."; }
        printf '%s\n' "$p/lib" >> "$LIBFILE.tmp"
    done
    mv -f "$LIBFILE.tmp" "$LIBFILE"
fi
mapfile -t LIBDIRS < "$LIBFILE"
GLIBC_LIB="${LIBDIRS[0]}"
LOADER="$GLIBC_LIB/ld-linux-x86-64.so.2"
[ -x "$LOADER" ] || die "glibc loader not found at $LOADER (delete $LIBFILE and retry)."

LIBPATH=$(IFS=:; echo "${LIBDIRS[*]}")
# System GPU drivers (hardware.graphics.enable / hardware.opengl.enable). Mesa, NVIDIA, Vulkan ICDs live here.
if [ -d /run/opengl-driver/lib ]; then
    LIBPATH="$LIBPATH:/run/opengl-driver/lib"
else
    say "warning: /run/opengl-driver/lib not found - GPU drivers may be missing (set hardware.graphics.enable = true)."
fi
LIBPATH="$LIBPATH:$HERE"

# ---------------------------------------------------------------- environment for the app
export ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-$HERE/libonnxruntime.so.1.31.0}"
export COMPCLASS_MODEL_DIR="${COMPCLASS_MODEL_DIR:-$HERE/exports/final}"
export COMPCLASS_DEMO_DIR="${COMPCLASS_DEMO_DIR:-$HERE/demo_stack}"
# Cursor themes on NixOS live in the profiles, not /usr/share/icons.
export XCURSOR_PATH="${XCURSOR_PATH:-$HOME/.icons:$HOME/.local/share/icons:$HOME/.nix-profile/share/icons:/etc/profiles/per-user/${USER:-$(id -un)}/share/icons:/run/current-system/sw/share/icons}"

HEADLESS=0
case "${ARGS[0]:-}" in classify | preprocess | models | -h | --help | -V | --version) HEADLESS=1 ;; esac

if [ "$HEADLESS" = 0 ] && [ "$DOCTOR" = 0 ]; then
    if [ "$USE_X11" = 1 ]; then
        [ -n "${DISPLAY:-}" ] || die "--x11 needs an X server (DISPLAY is empty). On niri start xwayland-satellite first, e.g.:
    xwayland-satellite :12 &  DISPLAY=:12 $0 --x11"
        unset WAYLAND_DISPLAY WAYLAND_SOCKET
    else
        if [ -z "${WAYLAND_DISPLAY:-}" ]; then
            # Inside niri WAYLAND_DISPLAY is normally set; recover it if the terminal lost it.
            rt="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
            sock=""
            for s in "$rt"/wayland-[0-9]*; do
                case "$s" in *.lock) continue ;; esac
                [ -S "$s" ] && { sock="$s"; break; }
            done
            [ -n "$sock" ] || die "no Wayland session found (WAYLAND_DISPLAY empty, no socket in $rt).
  Start this from a terminal inside niri, or test headless: $0 classify demo_stack/01_battery_holder.jpg"
            WAYLAND_DISPLAY="${sock##*/}"
            export XDG_RUNTIME_DIR="$rt" WAYLAND_DISPLAY
        fi
        # Force native Wayland: never let GPUI fall back to an X server that niri does not have.
        unset DISPLAY
    fi
fi

if [ "$DOCTOR" = 1 ]; then
    echo "bundle        $HERE"
    echo "loader        $LOADER"
    echo "library path  $LIBPATH" | tr ':' '\n' | sed '2,$s/^/              /'
    echo "WAYLAND       ${WAYLAND_DISPLAY:-<unset>}   DISPLAY ${DISPLAY:-<unset>}   XDG_RUNTIME_DIR ${XDG_RUNTIME_DIR:-<unset>}"
    for l in libwayland-client.so.0 libxkbcommon.so.0 libxcb.so.1 libEGL.so.1 libvulkan.so.1 libstdc++.so.6; do
        f=""; IFS=: read -ra D <<< "$LIBPATH"; for d in "${D[@]}"; do [ -e "$d/$l" ] && { f="$d/$l"; break; }; done
        printf '  %-24s %s\n' "$l" "${f:-NOT FOUND}"
    done
    ls /run/opengl-driver/share/vulkan/icd.d 2>/dev/null | sed 's/^/  vulkan ICD  /' || true
    exec "$LOADER" --library-path "$LIBPATH" "$BIN" models
fi

# The app is started through the Nix glibc loader with an explicit library path (it applies to the whole process,
# including onnxruntime and the Wayland/EGL/Vulkan libraries loaded with dlopen). Nothing is changed on the system,
# and programs the app launches (e.g. a browser for licence links) do not inherit the path.
exec "$LOADER" --library-path "$LIBPATH" "$BIN" "${ARGS[@]}"
