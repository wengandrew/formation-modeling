"""Repository paths, resolved from this file rather than the working directory."""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts):
    """Absolute path to a location inside the repository."""
    return os.path.join(REPO_ROOT, *parts)


def relative(path):
    """Path relative to the repository root, for messages."""
    return os.path.relpath(path, REPO_ROOT)
