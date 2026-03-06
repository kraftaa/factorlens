import os
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_cwd() -> Path:
    explicit = os.environ.get("FACTORLENS_CWD")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path.cwd()


def _packaged_binary() -> str | None:
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "bin" / "factorlens",
        pkg_dir / "bin" / "factorlens.exe",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return None


def _find_repo_root(start: Path) -> Path | None:
    env_root = os.environ.get("FACTORLENS_REPO")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if (p / "Cargo.toml").exists():
            return p

    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "Cargo.toml").exists() and (p / "crates" / "factor_cli").exists():
            return p
    return None


def _find_factorlens_cmd(cwd: Path) -> tuple[list[str], Path | None]:
    explicit = os.environ.get("FACTORLENS_BIN")
    if explicit:
        return [explicit], cwd

    packaged = _packaged_binary()
    if packaged:
        return [packaged], cwd

    on_path = shutil.which("factorlens")
    if on_path:
        return [on_path], cwd

    cargo = shutil.which("cargo")
    repo = _find_repo_root(cwd)
    if cargo and repo:
        return [cargo, "run", "-p", "factor_cli", "--"], repo

    raise FileNotFoundError(
        "Could not find runnable FactorLens backend. Install a wheel that bundles the binary, "
        "set FACTORLENS_BIN, or run inside a cloned FactorLens repo with Rust/cargo available."
    )


def main() -> int:
    argv = sys.argv[1:]
    cwd = _resolve_cwd()
    cmd, cmd_cwd = _find_factorlens_cmd(cwd)

    env = os.environ.copy()
    proc = subprocess.run(cmd + argv, cwd=str(cmd_cwd or cwd), env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
