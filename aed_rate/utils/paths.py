"""
Location of the generated data artefacts (precomputed coupling grids, PECs).

Coupling grids are expensive to produce and are cached as .npz files rather
than recomputed.  They are generated output, not source, so they live in a
``data/`` directory that is excluded from version control.

The directory is resolved relative to the current working directory, matching
how the scripts have always been run (from the repository root).  Override it
with the ``AED_DATA_DIR`` environment variable to keep the cache elsewhere.
"""

import os
from pathlib import Path

DATA_DIR_ENV_VAR = "AED_DATA_DIR"
DEFAULT_DATA_DIR = "data"


def data_dir(create: bool = False) -> Path:
    """
    Return the directory holding generated .npz artefacts.

    Parameters
    ----------
    create : bool
        Create the directory (and any parents) if it does not exist.

    Returns
    -------
    Path
        ``$AED_DATA_DIR`` if set, otherwise ``./data``.
    """
    path = Path(os.environ.get(DATA_DIR_ENV_VAR, DEFAULT_DATA_DIR))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def data_file(name: str, create_dir: bool = False) -> Path:
    """
    Return the full path of a named artefact inside the data directory.

    A name that already contains a directory component is returned unchanged,
    so callers can still pass an explicit path (e.g. a command-line argument).

    Parameters
    ----------
    name : str
        Bare file name, e.g. ``'oh_minus_coupling_6-31g.npz'``.
    create_dir : bool
        Create the data directory if it does not exist.

    Returns
    -------
    Path
        Path to the artefact.
    """
    candidate = Path(name)
    if candidate.parent != Path("."):
        return candidate
    return data_dir(create=create_dir) / candidate.name
