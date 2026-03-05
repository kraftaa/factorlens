import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_factorlens_bin() -> list[str]:
    explicit = os.environ.get("FACTORLENS_BIN")
    if explicit:
        return [explicit]

    on_path = shutil.which("factorlens")
    if on_path:
        return [on_path]

    cargo = shutil.which("cargo")
    if cargo:
        # Dev fallback: runs workspace binary directly.
        return [cargo, "run", "-p", "factor_cli", "--"]

    raise FileNotFoundError(
        "Could not find factorlens binary. Install Rust CLI first or set FACTORLENS_BIN."
    )


def main() -> int:
    argv = sys.argv[1:]
    cmd = _find_factorlens_bin() + argv

    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=_repo_root(), env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
