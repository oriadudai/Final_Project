"""One-time PTB-XL download for training the diagnostic-consistency classifier.

PTB-XL (Wagner et al. 2020, \\cite{ptbxl2020}) provides ~21,800 12-lead ECGs
with clinician-validated diagnostic labels (SCP codes), sampled at both 100 Hz
and 500 Hz. We only need the 100 Hz records (smaller download, ~1.7 GB) since
they are downsampled to FS=125 Hz at training time anyway.

Usage (run once, before scripts/train_diagnostic_classifier.py):
    python scripts/download_ptbxl.py [--dl-dir data/PTBXL]

The script is idempotent — wfdb skips files that already exist locally.
"""

import argparse
import os

import wfdb


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PTB-XL (100 Hz records + metadata)")
    parser.add_argument("--dl-dir", type=str, default=os.path.join("data", "PTBXL"),
                        help="Local directory to download PTB-XL into.")
    args = parser.parse_args()

    os.makedirs(args.dl_dir, exist_ok=True)

    print(f"Downloading PTB-XL into {args.dl_dir} ...")
    print("This pulls the 100 Hz waveform records plus metadata CSVs "
          "(ptbxl_database.csv, scp_statements.csv) — roughly 1.7 GB.")

    # PTB-XL is versioned on PhysioNet; "ptb-xl" resolves to the latest release.
    wfdb.dl_database(
        "ptb-xl",
        dl_dir=args.dl_dir,
        keep_subdirs=True,
        overwrite=False,
    )

    print("\nDone. Expected layout:")
    print(f"  {args.dl_dir}/ptbxl_database.csv")
    print(f"  {args.dl_dir}/scp_statements.csv")
    print(f"  {args.dl_dir}/records100/00000/00001_lr.dat / .hea  ...")
    print("\nNext: python scripts/train_diagnostic_classifier.py")


if __name__ == "__main__":
    main()
