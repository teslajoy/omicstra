"""omicstra - cross-modal biomedical embedding alignment + reasoning via MCP."""

from importlib.metadata import PackageNotFoundError, version

try:
    # read from the installed distribution rather than restating it here. a
    # literal drifted from pyproject the moment 1.0.0 was tagged, because
    # nothing imported it and so nothing failed.
    __version__ = version("omicstra")
except PackageNotFoundError:          # running from a source tree, not installed
    __version__ = "0+unknown"

__all__ = ["__version__"]
