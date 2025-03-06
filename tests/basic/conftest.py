"""
Pytest configuration file for basic tests.
Automatically imports patch_commands to ensure necessary stubs are in place.
"""

import pytest

# Import the module that applies the monkey patches
import tests.basic.patch_commands


def pytest_configure(config):
    """Configure pytest before running tests.
    This runs before any tests are executed.
    """
    # The import above already applies the patches
    pass 