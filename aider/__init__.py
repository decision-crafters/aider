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

# Ensure taskmanager module is accessible
try:
    from . import taskmanager
except ImportError:
    pass

from .taskmanager import Task, Environment, TestInfo, TaskManager, get_task_manager

__all__ = [__version__, "taskmanager", 'Task', 'Environment', 'TestInfo', 'TaskManager', 'get_task_manager']
