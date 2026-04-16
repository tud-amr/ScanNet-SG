import os
import sys
from pathlib import Path


class ThirdPartyError(RuntimeError):
    pass


def thirdparty_dir() -> Path:
    """
    Canonical location for third-party source trees in this project.
    """
    return Path(__file__).resolve().parent


def ensure_submodule(
    *,
    name: str,
    dest_dir: Path,
) -> Path:
    """
    Ensure a git submodule checkout exists at dest_dir and return its path.

    Third-party projects in this repo are provided via git submodules; we do not
    download them at runtime.
    """
    # A checked-out submodule typically has a `.git` *file* (gitdir: ...) rather than a
    # `.git` directory, so check `.exists()` instead of `.is_dir()`.
    if dest_dir.is_dir() and (dest_dir / ".git").exists():
        return dest_dir

    if dest_dir.exists() and not dest_dir.is_dir():
        raise ThirdPartyError(f"{name} exists but is not a directory: {dest_dir}")

    raise ThirdPartyError(
        f"{name} is not checked out at:\n"
        f"  {dest_dir}\n\n"
        f"Initialize submodules and retry:\n"
        f"  git submodule sync --recursive\n"
        f"  git submodule update --init --recursive\n"
    )


def ensure_grounded_sam(*, from_file: str | os.PathLike | None = None) -> Path:
    """
    Ensure Grounded-Segment-Anything source tree exists and return its path.
    """
    tp = thirdparty_dir()
    dest = tp / "Grounded-Segment-Anything"
    return ensure_submodule(name="Grounded-Segment-Anything", dest_dir=dest)


def ensure_recognize_anything(*, from_file: str | os.PathLike | None = None) -> Path:
    """
    Ensure recognize-anything source tree exists and return its path.
    """
    tp = thirdparty_dir()
    dest = tp / "recognize-anything"
    return ensure_submodule(name="recognize-anything", dest_dir=dest)


def add_to_syspath(path: str | os.PathLike) -> None:
    p = str(Path(path).resolve())
    if p not in sys.path:
        # Append (not prepend) to avoid shadowing local modules.
        sys.path.append(p)


def prepend_to_syspath(path: str | os.PathLike) -> None:
    """Put *path* first on sys.path so it wins over other installs (e.g. PYTHONPATH)."""
    p = str(Path(path).resolve())
    try:
        sys.path.remove(p)
    except ValueError:
        pass
    sys.path.insert(0, p)

