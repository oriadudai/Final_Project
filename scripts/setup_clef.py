"""One-time CLEF setup script.

Clones the Nokia Bell Labs CLEF repository, applies the Windows UTF-8
encoding patch to setup.py, and installs the package into the active
Python environment.

Usage (run once after creating the conda environment):
    python scripts/setup_clef.py

The script is idempotent — safe to re-run if something fails midway.
"""

import os
import subprocess
import sys

CLEF_REPO_URL = "https://github.com/Nokia-Bell-Labs/ecg-foundation-model.git"
CLEF_LOCAL_DIR = os.path.join("models", "clef_repo")


def run(cmd: list, **kwargs) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def patch_setup_py(setup_path: str) -> None:
    """Fix encoding issue in setup.py (open README.md without utf-8 on Windows)."""
    with open(setup_path, "r", encoding="utf-8") as f:
        content = f.read()

    patched = content.replace(
        'with open("README.md", "r") as fh:',
        'with open("README.md", "r", encoding="utf-8") as fh:',
    )

    if patched == content:
        print("  setup.py already patched or pattern not found — skipping.")
        return

    with open(setup_path, "w", encoding="utf-8") as f:
        f.write(patched)
    print("  Patched setup.py to use UTF-8 when reading README.md.")


def main() -> None:
    # Ensure models/ directory exists
    os.makedirs("models", exist_ok=True)

    # 1. Clone the repo (shallow, only latest commit)
    if os.path.isdir(CLEF_LOCAL_DIR):
        print(f"[1/3] {CLEF_LOCAL_DIR} already exists — skipping clone.")
    else:
        print(f"[1/3] Cloning CLEF from {CLEF_REPO_URL} ...")
        run(["git", "clone", "--depth", "1", CLEF_REPO_URL, CLEF_LOCAL_DIR])

    # 2. Patch setup.py for Windows encoding compatibility
    setup_path = os.path.join(CLEF_LOCAL_DIR, "setup.py")
    print("[2/3] Applying Windows UTF-8 patch to setup.py ...")
    patch_setup_py(setup_path)

    # 3. Install in editable mode so all subpackages (baselines/, etc.) are importable.
    # --no-deps prevents CLEF from downgrading torch to its pinned version.
    print("[3/3] Installing CLEF (editable, no-deps) ...")
    run([sys.executable, "-m", "pip", "install", "-e", CLEF_LOCAL_DIR, "--no-deps"])

    print("\nCLEF setup complete.")
    print("Next: download a pretrained checkpoint (see README Step 0b).")


if __name__ == "__main__":
    main()
