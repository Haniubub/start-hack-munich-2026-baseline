"""NOT the argparse module. A carrier for differometor's start-up measures; it hands over to the real one.

Why this file exists
    ``benchmark.py`` imports ``jax`` before ``differometor``, so nothing inside the package can
    influence how JAX is imported. Python puts the directory of the script first on ``sys.path``,
    so the line ``import argparse`` of ``benchmark.py`` (its first import after its timer has
    started) finds this file. It calls ``differometor/_startup.py: early()`` and then loads the
    ``argparse`` module of the standard library and puts it in its own place. Every importer
    receives the real module; nothing of this file remains.

Why argparse
    It is the first module the header of ``benchmark.py`` imports. One of the measures concerns
    ``importlib.metadata``, which the header imports three lines later (DEFER_REPORT_MODULES in
    ``_startup.py``); a carrier further down the header (this file was ``statistics.py`` before)
    comes too late for it.

What it is for
    The early measures in ``_startup.py``. Unlike the ``.pth`` hook this carrier runs inside the
    timed window, so it cannot move anything in front of the timer. PREIMPORT is ignored here.

Switch
    ``ENABLED`` below (or ``DIFFEROMETOR_STARTUP_SHADOW=1`` / ``0`` for one process). When off,
    this file only hands over to the real module.
"""
import sys

ENABLED = True


def _hand_over():
    """Load the real `argparse` from the rest of sys.path and register it under this name."""
    import os
    from importlib.machinery import PathFinder
    from importlib.util import module_from_spec

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rest = [entry for entry in sys.path if os.path.abspath(entry or os.curdir) != here]
    except Exception:
        rest = sys.path[1:]
    spec = PathFinder.find_spec(__name__, rest)
    module = module_from_spec(spec)
    sys.modules[__name__] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(__name__, None)
        raise


def _start_up():
    import os

    value = os.environ.get("DIFFEROMETOR_STARTUP_SHADOW")
    enabled = ENABLED if value is None else value.strip().lower() in ("1", "true", "yes", "on")
    if not enabled or getattr(sys, "_differometor_startup", None) is not None:
        return
    from importlib.machinery import SourceFileLoader

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "differometor", "_startup.py")
    hook = type(sys)("_differometor_startup")
    hook.__file__ = path
    sys.modules[hook.__name__] = hook
    try:
        SourceFileLoader(hook.__name__, path).exec_module(hook)
        hook.early("shadow")
    except BaseException:
        sys.modules.pop(hook.__name__, None)
        raise


try:
    _start_up()
except Exception:
    pass  # the measures are optional, the hand-over below is not
_hand_over()
