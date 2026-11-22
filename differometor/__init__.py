from differometor import _tuning

# Cheaper XLA compilation and JAX lowering for the whole process, one switch per measure in
# _tuning.py. Has to run before the first program is compiled.
_tuning.apply()

from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector

__all__ = [
    "Setup",
    "run",
    "run_build_step",
    "run_simulation_step",
    "simulate",
    "visualize_setup",
    "power_detector",
    "signal_detector",
]


def __getattr__(name):
    # plot.py pulls in plotly, which a plain simulation never needs,
    # so it is only imported the first time someone asks for it.
    if name == "visualize_setup":
        from differometor.plot import visualize_setup

        globals()["visualize_setup"] = visualize_setup
        return visualize_setup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"visualize_setup"})


# Collector policy for the run (and bookkeeping of the start-up measures), see _startup.py.
# When the carrier (argparse.py) has run, it has executed this very _startup.py already, as the
# module `_differometor_startup`. That module is then `differometor._startup` as well, instead of
# executing the file a second time (0.6 ms). Without the carrier the file is imported as usual.
import os as _os
import sys as _sys

_hook = _sys.modules.get("_differometor_startup")
_file = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_startup.py")
if _hook is not None and _os.path.abspath(getattr(_hook, "__file__", None) or "") == _file and hasattr(_hook, "late"):
    if getattr(_hook, "__spec__", None) is None:  # what importlib.reload asks for
        from importlib.machinery import ModuleSpec as _ModuleSpec, SourceFileLoader as _SourceFileLoader

        _hook.__spec__ = _ModuleSpec(__name__ + "._startup", _SourceFileLoader(__name__ + "._startup", _file), origin=_file)
        del _ModuleSpec, _SourceFileLoader
    _sys.modules[__name__ + "._startup"] = _startup = _hook
else:
    from differometor import _startup as _startup
del _os, _sys, _file, _hook

_startup.late()
