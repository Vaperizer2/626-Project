#!/usr/bin/env python3
"""Package an already-built Windows x64 executable, never graphs or models.

Uses only the Python standard library. The source commit is declared provenance,
not inferred from the current checkout: package only the matching final build.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import struct
import tempfile
import zipfile


REPO = Path(__file__).resolve().parents[3]
PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
DEFAULT_OUTPUT = REPO / "dist/local-main/component-classifier-windows-fix1.zip"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def windows_text(text: str) -> bytes:
    return text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")


def check_windows_x64(exe: bytes) -> None:
    """Catch accidentally selecting a Linux binary or the wrong Windows target."""
    if len(exe) < 64 or exe[:2] != b"MZ":
        raise ValueError("input is not a Windows PE executable (missing DOS header)")
    offset = struct.unpack_from("<I", exe, 0x3C)[0]
    if offset + 6 > len(exe) or exe[offset : offset + 4] != b"PE\0\0":
        raise ValueError("input is not a Windows PE executable (missing PE header)")
    if struct.unpack_from("<H", exe, offset + 4)[0] != 0x8664:
        raise ValueError("input is not an x86_64 Windows executable")


def package(exe_path: Path, source_commit: str, output: Path, *, force: bool = False) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit):
        raise ValueError("--source-commit must be the full 40-character source commit hash")
    checksum_path = output.with_name(output.name + ".sha256")
    if exe_path.resolve() in (output.resolve(), checksum_path.resolve()):
        raise ValueError("output must not overwrite the input executable")
    if not force and (output.exists() or checksum_path.exists()):
        raise ValueError("output or checksum already exists; choose --output or use --force")
    exe = exe_path.read_bytes()
    check_windows_x64(exe)
    files = {
        "gpui-demo.exe": exe,
        "run.bat": windows_text((PACKAGING / "run.bat").read_text(encoding="utf-8")),
        "UPDATE_WINDOWS_FIX1.txt": windows_text(
            (PACKAGING / "UPDATE_WINDOWS_FIX1.txt").read_text(encoding="utf-8")
        ),
        "BUILD_INFO_FIX1.txt": windows_text(
            "Component Classifier Windows fix1: application-only overlay\n"
            f"Source commit (declared by packager): {source_commit.lower()}\n"
            f"Executable SHA-256: {sha256(exe)}\n"
            f"Executable bytes: {len(exe)}\n"
            "Target: x86_64-pc-windows-msvc\n"
            "Build: cargo xwin build --release --locked --target x86_64-pc-windows-msvc "
            "-p gpui-demo --features turbojpeg-vendored\n"
            "CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true\n"
            "Requires the local gpui-pre-windows patch embedding HLSL source.\n"
            "Preserves the original bundle's CPU ONNX Runtime 1.31.0 and all models.\n"
            "Native Windows GUI validation: PENDING. Packaging does not run the binary.\n"
            "See UPDATE_WINDOWS_FIX1.txt for installation and checksum scope.\n"
        ),
    }
    # A flat, explicit allowlist: no recursive bundle copy and no model inputs.
    files["SHA256SUMS-WINDOWS-FIX1.txt"] = (
        "".join(f"{sha256(data)}  {name}\n" for name, data in sorted(files.items())).encode("ascii")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Fixed timestamps, modes and order make equal inputs reproducible (same zlib).
    with tempfile.TemporaryDirectory(prefix=".windows-fix1-", dir=output.parent) as work:
        archive = Path(work) / "update.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for name, data in sorted(files.items()):
                entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(entry, data)
        checksum = Path(work) / "update.sha256"
        checksum.write_text(f"{sha256(archive.read_bytes())}  {output.name}\n", encoding="ascii")
        archive.replace(output)
        checksum.replace(checksum_path)
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")
    print(f"Wrote {checksum_path}")
    print("Native Windows GUI validation remains pending.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True, help="new, already-built Windows x64 gpui-demo.exe")
    parser.add_argument("--source-commit", required=True, help="full source commit for that executable")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output ZIP (default: %(default)s)")
    parser.add_argument("--force", action="store_true", help="replace an existing ZIP and checksum")
    args = parser.parse_args()
    try:
        package(args.exe, args.source_commit, args.output, force=args.force)
    except (OSError, ValueError) as error:
        parser.exit(1, f"package_windows_update: {error}\n")


if __name__ == "__main__":
    main()
