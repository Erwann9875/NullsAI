from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from elftools.elf.elffile import ELFFile


def verify(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        elf = ELFFile(stream)
        if elf.elfclass != 64 or not elf.little_endian or elf["e_machine"] != "EM_X86_64":
            raise ValueError("external reader is not a little-endian x86-64 ELF")
        if any(segment["p_type"] == "PT_INTERP" for segment in elf.iter_segments()):
            raise ValueError("external reader is dynamically linked")
        dynamic = elf.get_section_by_name(".dynamic")
        if dynamic is not None and any(tag.entry.d_tag == "DT_NEEDED" for tag in dynamic.iter_tags()):
            raise ValueError("external reader has dynamic library dependencies")
    data = path.read_bytes()
    for marker in (
        b"nullsroyale-external-reader.v1",
        b"nullsroyale.rel.free",
        b"nullsroyale-owner-ladder-2026-11-23_2026-12-01",
    ):
        if marker not in data:
            raise ValueError(f"external reader is missing marker {marker!r}")
    return {
        "ok": True,
        "schema": "nullsroyale-external-reader-build.v1",
        "architecture": "x86_64",
        "static": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "output": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.binary), sort_keys=True))


if __name__ == "__main__":
    main()
