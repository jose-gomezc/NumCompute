"""Shared pytest setup: draw figures off-screen and close them after each test."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen first)
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
