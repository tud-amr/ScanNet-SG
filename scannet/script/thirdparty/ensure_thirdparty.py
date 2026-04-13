import os
import subprocess
import sys
from pathlib import Path


class ThirdPartyError(RuntimeError):
    pass


def thirdparty_dir() -> Path:
    """
    Canonical location for optional third-party source trees in this project.
    """
    return Path(__file__).resolve().parent


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    try:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
    except FileNotFoundError as e:
        raise ThirdPartyError(
            f"Missing executable: {cmd[0]!r}. Please install it and retry."
        ) from e
    except subprocess.CalledProcessError as e:
        raise ThirdPartyError(f"Command failed: {' '.join(cmd)}") from e


def ensure_git_repo(
    *,
    name: str,
    url: str,
    dest_dir: Path,
    recurse_submodules: bool = False,
    auto_clone: bool = True,
) -> Path:
    """
    Ensure a git repo exists at dest_dir. If missing and auto_clone=True, clone it.
    Returns dest_dir.
    """
    if dest_dir.is_dir() and (dest_dir / ".git").exists():
        return dest_dir

    if dest_dir.exists() and not dest_dir.is_dir():
        raise ThirdPartyError(f"{name} destination exists but is not a directory: {dest_dir}")

    if not auto_clone:
        raise ThirdPartyError(
            f"{name} is not available at {dest_dir}.\n"
            f"Clone it manually:\n"
            f"  git clone {'--recurse-submodules ' if recurse_submodules else ''}{url} {dest_dir}"
        )

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if recurse_submodules:
        cmd.append("--recurse-submodules")
    cmd += [url, str(dest_dir)]
    print(f"[thirdparty] Cloning {name} into {dest_dir} ...", file=sys.stderr)
    _run(cmd)
    return dest_dir


def ensure_grounded_sam(*, auto_clone: bool = True, from_file: str | os.PathLike | None = None) -> Path:
    """
    Ensure Grounded-Segment-Anything source tree exists and return its path.
    """
    tp = thirdparty_dir()
    dest = tp / "Grounded-Segment-Anything"
    return ensure_git_repo(
        name="Grounded-Segment-Anything",
        url="https://github.com/IDEA-Research/Grounded-Segment-Anything.git",
        dest_dir=dest,
        recurse_submodules=True,
        auto_clone=auto_clone,
    )


def ensure_recognize_anything(*, auto_clone: bool = True, from_file: str | os.PathLike | None = None) -> Path:
    """
    Ensure recognize-anything source tree exists and return its path.
    """
    tp = thirdparty_dir()
    dest = tp / "recognize-anything"
    return ensure_git_repo(
        name="recognize-anything",
        url="https://github.com/xinyu1205/recognize-anything.git",
        dest_dir=dest,
        recurse_submodules=False,
        auto_clone=auto_clone,
    )


def add_to_syspath(path: str | os.PathLike) -> None:
    p = str(Path(path).resolve())
    if p not in sys.path:
        # Append (not prepend) to avoid shadowing local modules.
        sys.path.append(p)

