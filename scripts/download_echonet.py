"""Download EchoNet-Dynamic from Redivis (Stanford AIMI).

Requires a Redivis API token, since the dataset sits behind the AIMI Research Use
Agreement. Create one at https://redivis.com/workspace/settings/tokens with at least
"Data access" scope, then:

    export REDIVIS_API_TOKEN=...
    python scripts/download_echonet.py

~7 GB. Resumes are not supported by the API, so it is written to a .part file and moved
into place only on success, which keeps a truncated download from being mistaken for a
complete one.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

TABLE = "aimi.echonet_dynamic:66s1:v1_0.echonet:fjdn"
FILENAME = "EchoNet-Dynamic.zip"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    if not os.environ.get("REDIVIS_API_TOKEN"):
        sys.exit(
            "REDIVIS_API_TOKEN is not set.\n"
            "  1. Go to https://redivis.com/workspace/settings/tokens\n"
            "  2. Create a token with data-access scope\n"
            "  3. export REDIVIS_API_TOKEN=<token>  then re-run\n"
        )
    import redivis

    DEST.mkdir(parents=True, exist_ok=True)
    final = DEST / FILENAME
    if final.exists() and final.stat().st_size > 1_000_000_000:
        print(f"[skip] {final} already present ({final.stat().st_size/1e9:.2f} GB)")
        return

    print(f"[table] {TABLE}")
    table = redivis.table(TABLE)

    try:
        listing = table.to_directory().list(recursive=True)
        print(f"[files] {len(listing)} entries in table")
        for f in listing[:10]:
            print("   ", f)
    except Exception as e:
        print(f"[warn] could not list table contents: {type(e).__name__}: {e}")

    part = DEST / (FILENAME + ".part")
    print(f"[get ] {FILENAME} -> {part}")
    t0 = time.time()
    f = table.file(FILENAME)
    f.download(str(part), overwrite=True)

    src = part if part.exists() else DEST / FILENAME
    # redivis may write the original filename into the directory rather than the exact path
    if src.is_dir():
        cand = next(iter(src.glob("*.zip")), None)
        if cand:
            src = cand
    size = src.stat().st_size
    if size < 1_000_000_000:
        sys.exit(f"[FAIL] downloaded only {size/1e6:.0f} MB, expected ~7 GB. Not moving into place.")
    src.replace(final)
    print(f"[ok  ] {final}  {size/1e9:.2f} GB in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
