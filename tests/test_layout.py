"""Verify package layout and imports."""

import importlib

import doppelt

SUBPACKAGES = (
    "doppelt.core",
    "doppelt.actions",
    "doppelt.engine",
    "doppelt.replay",
    "doppelt.sim",
    "doppelt.ml",
    "doppelt.cli",
)


def test_version():
    assert doppelt.__version__ == "0.1.0"


def test_subpackages_import():
    for name in SUBPACKAGES:
        module = importlib.import_module(name)
        assert module is not None
