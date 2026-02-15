"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def disable_gpu():
    """Disable GPU for tests to avoid CUDA compatibility issues."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
