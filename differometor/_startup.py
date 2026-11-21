"""Process-level start-up measures: what happens around `import jax` and with the garbage collector.

This file has two entry points and is deliberately standalone (standard library only, nothing
from differometor, jax or numpy at module level), because the first entry point runs before any
of those can be imported:

``early()``
    Has to run *before* ``import jax`` to have any effect. `differometor` itself is imported
    after `jax` by the benchmark, so `early()` needs one of two carriers:

    * the start-up hook ``differometor_startup.pth`` (installed into site-packages by
      ``pip install -e .`` when the opt-in in ``setup.py`` is on). It runs at interpreter
      start, i.e. before the first timer of ``benchmark.py``;
    * the module ``statistics.py`` next to ``benchmark.py`` (see there). It runs inside the
      timed window, at the ``import statistics`` line of the benchmark.

    Without a carrier `early()` never runs and nothing below marked "early" happens.

``late()``
    Called at the end of ``differometor/__init__.py``. Needs no carrier.

Every measure has a switch below. Two kinds of measure are kept strictly apart:

kind (A)  removes work from the process (collector passes that find nothing, modules that the
          run never touches). Nothing is computed ahead of time.
kind (B)  does NOT remove work. It moves the imports in front of the first timer of
          ``benchmark.py`` ("interpreter launch", which the benchmark excludes). Off by default.
          Only possible with the ``.pth`` carrier.

For measurements the switches can be overridden per process without editing this file:
``DIFFEROMETOR_STARTUP`` is a comma separated list out of ``gc``, ``lazy``, ``preimport``
(or ``none``) and replaces the three early switches; ``DIFFEROMETOR_GC`` replaces ``GC_RUN``.
"""
import sys

# --------------------------------------------------------------------------------------------
# Switches
# --------------------------------------------------------------------------------------------

# early, kind (A): the cyclic garbage collector does not run while `import jax` executes.
# Importing JAX allocates about 110 000 long-lived objects (functions, classes, dicts) and no
# garbage worth mentioning; the collector nevertheless runs about 150 times over them. When the
# import is done, the objects are frozen (moved to the permanent generation, so no later pass
# has to look at them either) and the collector is switched back on.
GC_PAUSE_JAX_IMPORT = True

# early, kind (A): the JAX modules in LAZY_MODULES are created empty and executed on first use
# (importlib.util.LazyLoader). `import jax` pulls them in, this workload never touches them.
# Anything that does touch them later gets the real module, just later.
LAZY_JAX_MODULES = True

# early, kind (B): MOVES WORK OUTSIDE THE TIMED WINDOW. When the process is `benchmark.py`, the
# hook repeats the environment set-up of its header (JAX_PLATFORMS, JAX_ENABLE_X64, CPU affinity,
# thread limits) and imports everything the header imports, before the benchmark starts its
# timer. The timed imports then find all modules in `sys.modules`. Same work, not measured.
PREIMPORT = False

# late, kind (A): what the collector does after `differometor` has been imported.
#   "default"  leave it alone
#   "relaxed"  freeze everything that exists now and raise the thresholds (GC_RELAXED_THRESHOLDS):
#              the collector keeps working, but it looks at young objects far less often
#   "off"      freeze and gc.disable(): no collector passes for the rest of the process
GC_RUN = "relaxed"
GC_RELAXED_THRESHOLDS = (50_000, 20, 100)

# Modules that `import jax` (0.9.0) loads and that a plain CPU run never touches. Found by
# loading all of jax and jaxlib lazily and listing what was still unloaded after the timed
# phases of the benchmark. A name that does not exist in another JAX version is simply never asked for.
LAZY_MODULES = frozenset({
    # public namespaces that only re-export; the implementation behind them is loaded with them
    "jax.api_util", "jax.custom_batching", "jax.custom_transpose", "jax.debug", "jax.distributed",
    "jax.dlpack", "jax.dtypes", "jax.errors", "jax.export", "jax.ffi", "jax.image",
    "jax.interpreters.ad", "jax.interpreters.batching", "jax.interpreters.mlir",
    "jax.interpreters.partial_eval", "jax.interpreters.pxla", "jax.interpreters.xla",
    "jax.lax", "jax.lib", "jax.memory", "jax.monitoring", "jax.nn", "jax.numpy.fft",
    "jax.numpy.linalg", "jax.ops", "jax.profiler", "jax.random", "jax.scipy", "jax.sharding",
    "jax.stages", "jax.tree", "jax.typing",
    "jax.experimental.compilation_cache.compilation_cache", "jax.experimental.multihost_utils",
    # internals nobody asks for on one CPU
    "jax._src.clusters", "jax._src.error_check", "jax._src.mesh_utils", "jax._src.numpy.linalg",
    "jax._src.pickle_util",
    # accelerator and LAPACK bindings of jaxlib
    "jaxlib.cpu_sparse", "jaxlib.gpu_linalg", "jaxlib.gpu_prng", "jaxlib.gpu_rnn",
    "jaxlib.gpu_solver", "jaxlib.gpu_sparse", "jaxlib.gpu_triton", "jaxlib.lapack",
    "jaxlib.mlir.dialects.builtin", "jaxlib.mosaic.python.mosaic_gpu", "jaxlib.mosaic.python.tpu",
})

# What the header of benchmark.py imports (lines 6 to 40), in its order. Used by PREIMPORT only.
_BENCHMARK_HEADER = (
    "argparse", "datetime", "hashlib", "importlib.metadata", "json", "os", "pathlib", "platform",
    "statistics", "subprocess", "jax", "jax.numpy", "numpy", "differometor", "differometor.setups",
)

# Filled in by early() and late(): which measures are active in this process, and through
# which carrier. Shared between the two possible copies of this module through `sys`.
_STATE_ATTRIBUTE = "_differometor_startup"


def state():
    """What the start-up measures did in this process (a dict, empty when nothing ran)."""
    return getattr(sys, _STATE_ATTRIBUTE, None) or {}


def _early_options():
    import os

    value = os.environ.get("DIFFEROMETOR_STARTUP")
    if value is None:
        return {
            name
            for name, on in (("gc", GC_PAUSE_JAX_IMPORT), ("lazy", LAZY_JAX_MODULES), ("preimport", PREIMPORT))
            if on
        }
    return {part.strip().lower() for part in value.split(",")} & {"gc", "lazy", "preimport"}


class _PausedCollector:
    """Loader wrapper for the top-level `jax` package: no collector passes while it executes."""

    def __init__(self, loader, watcher):
        self._loader = loader
        self._watcher = watcher

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        import gc

        # Put the real loader back first: nothing outside ever sees this wrapper.
        try:
            module.__spec__.loader = self._loader
            module.__loader__ = self._loader
        except Exception:
            pass
        pause = self._watcher.pause and gc.isenabled()
        if pause:
            gc.disable()
        try:
            self._loader.exec_module(module)
        finally:
            self._watcher.remove()
            if pause:
                # Everything allocated meanwhile is module-level state that lives as long as
                # the process. Freezing it keeps later passes from walking it once more.
                gc.freeze()
                gc.enable()
                state()["gc_paused_for_jax"] = True

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _Watcher:
    """Entry on `sys.meta_path` that waits for the import of `jax` and removes itself afterwards.

    It never loads anything itself. For `jax` it hands back the normal spec with a loader that
    pauses the collector, for names in `lazy` the normal spec with a lazy loader, and for every
    other name `None`, which sends the import machinery on to the normal finders.
    """

    def __init__(self, pause, lazy):
        self.pause = pause
        self.lazy = lazy
        self._busy = False

    def remove(self):
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass

    def find_spec(self, name, path=None, target=None):
        if name != "jax" and name not in self.lazy:
            return None
        if self._busy:
            return None
        self._busy = True
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(name, path, target)
                if spec is not None:
                    break
            if spec is None or spec.loader is None:
                return None
            if name == "jax":
                spec.loader = _PausedCollector(spec.loader, self)
            else:
                from importlib.machinery import SourceFileLoader

                if type(spec.loader) is not SourceFileLoader:
                    return None
                from importlib.util import LazyLoader

                spec.loader = LazyLoader(spec.loader)
                state().setdefault("lazy", []).append(name)
            return spec
        except Exception:
            # Never stand in the way of an import: the normal finders take over.
            return None
        finally:
            self._busy = False


def _is_benchmark_process():
    import os

    argv = getattr(sys, "argv", None) or [""]
    return os.path.basename(argv[0]) == "benchmark.py"


def _preimport():
    """Kind (B): import what the header of benchmark.py imports, before its timer starts.

    The header prepares the process before it imports the numerical libraries (platform and
    precision for JAX, CPU affinity and thread limits before the libraries start workers). The
    same preparation has to happen here first, otherwise the libraries would be initialized
    differently than in an ordinary run. Only done for a process whose script is benchmark.py.
    """
    import importlib
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ.setdefault("MPLBACKEND", "Agg")
    cpus = None
    argv = sys.argv[1:]
    for index, argument in enumerate(argv):
        if argument == "--cpus" and index + 1 < len(argv):
            cpus = int(argv[index + 1])
        elif argument.startswith("--cpus="):
            cpus = int(argument.split("=", 1)[1])
    if cpus is not None:
        allowed = sorted(os.sched_getaffinity(0))
        if not 1 <= cpus <= len(allowed):
            return False  # the benchmark reports the error itself
        os.sched_setaffinity(0, allowed[:cpus])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(cpus)
    # The script directory is put on sys.path after the hook has run. The benchmark insists that
    # differometor comes from the checkout next to it, so look there first, as the script will.
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    added = script_dir not in sys.path
    if added:
        sys.path.insert(0, script_dir)
    try:
        for name in _BENCHMARK_HEADER:
            importlib.import_module(name)
    finally:
        if added:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass
    return True


def early(carrier="pth"):
    """Apply the early measures. Safe to call more than once; only the first call acts."""
    if getattr(sys, _STATE_ATTRIBUTE, None) is not None:
        return state()
    info = {"carrier": carrier, "early": []}
    setattr(sys, _STATE_ATTRIBUTE, info)
    options = _early_options()
    if "jax" in sys.modules:
        # Too late for anything that concerns the import of JAX.
        return info
    pause = "gc" in options
    lazy = LAZY_MODULES if "lazy" in options else frozenset()
    if pause or lazy:
        sys.meta_path.insert(0, _Watcher(pause, lazy))
        info["early"] += [name for name in ("gc", "lazy") if name in options]
    if "preimport" in options and carrier == "pth" and _is_benchmark_process():
        if _preimport():
            info["early"].append("preimport")
    return info


def late():
    """Apply the collector policy for the run (GC_RUN). Called when differometor has been imported."""
    import gc
    import os

    info = getattr(sys, _STATE_ATTRIBUTE, None)
    if info is None:
        info = {"carrier": None, "early": []}
        setattr(sys, _STATE_ATTRIBUTE, info)
    if "late" in info:
        return info
    mode = os.environ.get("DIFFEROMETOR_GC", GC_RUN).strip().lower()
    info["late"] = mode
    if mode not in ("relaxed", "off") or not gc.isenabled():
        return info
    gc.freeze()
    if mode == "off":
        gc.disable()
    else:
        gc.set_threshold(*GC_RELAXED_THRESHOLDS)
    return info
