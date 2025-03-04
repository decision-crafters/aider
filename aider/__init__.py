from packaging import version

__version__ = "0.75.2.dev"
safe_version = __version__

try:
    from aider._version import __version__
except Exception:
    __version__ = safe_version + "+import"

if type(__version__) is not str:
    __version__ = safe_version + "+type"
else:
    try:
        if version.parse(__version__) < version.parse(safe_version):
            __version__ = safe_version + "+less"
    except Exception:
        __version__ = safe_version + "+parse"

# Make sure taskmanager is imported and accessible
# This ensures its classes are available for import from aider.taskmanager
from . import taskmanager

# Explicitly make these available at the package level
__all__ = ['__version__']
