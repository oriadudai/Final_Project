"""One-time PTB-XL download for training the diagnostic-consistency classifier.

Uses wget instead of wfdb.dl_database to avoid a URL-mangling bug in some
wfdb versions that concatenates records100 and records500 paths.

Downloads only the 100 Hz records (records100/, ~1.7 GB) and the two metadata
CSVs.  The 500 Hz records (~8.5 GB) are not needed.

Usage:
    python scripts/download_ptbxl.py [--dl-dir data/PTBXL]
"""

import argparse
import os
import subprocess
import sys


_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"
_METADATA = ["ptbxl_database.csv", "scp_statements.csv"]


def _wget(*args):
    cmd = ["wget"] + list(args)
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"wget failed (exit {ret.returncode}). Command: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(ret.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PTB-XL 100 Hz records + metadata")
    parser.add_argument("--dl-dir", type=str, default=os.path.join("data", "PTBXL"))
    args = parser.parse_args()

    os.makedirs(args.dl_dir, exist_ok=True)

    print("Downloading PTB-XL metadata CSVs ...")
    for fname in _METADATA:
        out = os.path.join(args.dl_dir, fname)
        if os.path.exists(out):
            print(f"  {fname} already exists, skipping.")
        else:
            _wget("-q", f"{_BASE_URL}/{fname}", "-O", out)
            print(f"  {fname} done.")

    print("\nDownloading records100/ (~1.7 GB — this takes a few minutes) ...")
    # -r  recursive   -N  timestamps (skip if up-to-date)   -c  continue partial
    # -np no-parent   -nH no host dir   --cut-dirs=3 strips /files/ptb-xl/1.0.3/
    # Result: data/PTBXL/records100/00000/00001_lr.dat  etc.
    _wget(
        "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3",
        "--show-progress", "-q",
        f"-P", args.dl_dir,
        f"{_BASE_URL}/records100/",
    )

    print(f"\nDone. Layout:")
    print(f"  {args.dl_dir}/ptbxl_database.csv")
    print(f"  {args.dl_dir}/scp_statements.csv")
    print(f"  {args.dl_dir}/records100/00000/00001_lr.dat ...")
    print(f"\nNext: python scripts/train_diagnostic_classifier.py --mode clef-probe")


if __name__ == "__main__":
    main()
