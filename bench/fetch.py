"""Download and verify the InterpBench artifacts for the six bench cases.

    .venv/bin/python -m bench.fetch            # download missing files, verify all
    .venv/bin/python -m bench.fetch --verify   # verify only

Files land in bench/interpbench/<case_id>/ and are checked against the SHA-256
recorded in bench/interpbench/manifest.json (which matches the Hugging Face LFS
object ids). The downloaded files are data, never imported or executed here.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "interpbench"
MANIFEST = ROOT / "manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="do not download; only check")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    failures = 0
    for case_id, entry in manifest["cases"].items():
        for name, info in entry["files"].items():
            path = ROOT / case_id / name
            if not path.exists():
                if args.verify:
                    print(f"{case_id}/{name}: MISSING")
                    failures += 1
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"{case_id}/{name}: downloading")
                urllib.request.urlretrieve(info["url"], path)
            digest = sha256(path)
            if digest != info["sha256"]:
                print(f"{case_id}/{name}: SHA-256 MISMATCH")
                failures += 1
            else:
                print(f"{case_id}/{name}: ok ({path.stat().st_size} bytes)")
    print("all verified" if failures == 0 else f"{failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
