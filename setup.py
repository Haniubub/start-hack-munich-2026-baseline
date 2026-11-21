"""Build script. Everything declarative is in pyproject.toml; this file only adds the optional start-up hook.

The start-up hook is one file, ``differometor_startup.pth``, that pip puts into site-packages next to
the package. Python executes it at every interpreter start in that environment, before any script runs.
It loads ``differometor/_startup.py`` (without importing differometor, jax or numpy) and calls ``early()``.
What that does, and which part of it removes work and which part only moves work in front of the
benchmark's timer, is described and switched in ``differometor/_startup.py``.

The hook is OFF unless asked for:

    STARTUP_HOOK = True below                                (the committed default for every install), or
    DIFFEROMETOR_STARTUP_HOOK=1 python -m pip install -e .   (for one install; 0 forces it off)

Remove it again with ``python -m pip uninstall differometor`` or by reinstalling with the opt-in off.
"""
import glob
import os

from setuptools import Command, setup
from setuptools.command.build import build

STARTUP_HOOK = False
HOOK_FILE = "packaging/differometor_startup.pth"


def startup_hook_requested():
    value = os.environ.get("DIFFEROMETOR_STARTUP_HOOK")
    if value is None:
        return STARTUP_HOOK
    return value.strip().lower() in ("1", "true", "yes", "on")


class build_startup_hook(Command):
    """Put differometor_startup.pth at the top level of the wheel, which pip unpacks into site-packages.

    A normal wheel takes everything from ``build_lib``. An editable wheel does not: setuptools assembles
    it in a temporary folder and only announces that folder as the ``install_lib`` of the ``install``
    command. If that convention ever changes, the folder check below fails and the hook is left out
    (with a warning); the package itself installs as usual.
    """

    description = "add the optional start-up hook (.pth) to the wheel"
    user_options = []
    editable_mode = False

    def initialize_options(self):
        self.build_lib = None

    def finalize_options(self):
        self.set_undefined_options("build", ("build_lib", "build_lib"))

    def _target(self):
        return os.path.join(self.build_lib, os.path.basename(HOOK_FILE))

    def run(self):
        if not startup_hook_requested():
            # A build folder left over from an earlier build with the hook must not leak it into this one.
            if not self.editable_mode and os.path.exists(self._target()):
                os.remove(self._target())
            return
        if self.editable_mode:
            folder = self.get_finalized_command("install").install_lib
            if not glob.glob(os.path.join(folder, "*.dist-info")):
                self.warn("start-up hook left out: cannot find the editable wheel under construction")
                return
            target = os.path.join(folder, os.path.basename(HOOK_FILE))
        else:
            target = self._target()
        self.mkpath(os.path.dirname(target))
        self.copy_file(HOOK_FILE, target, preserve_mode=False)

    # Protocol for build sub-commands (setuptools.command.build.SubCommand).
    def get_source_files(self):
        return [HOOK_FILE]

    def get_outputs(self):
        return [self._target()] if startup_hook_requested() and not self.editable_mode else []

    def get_output_mapping(self):
        return {}


class build_with_startup_hook(build):
    sub_commands = build.sub_commands + [("build_startup_hook", None)]


setup(cmdclass={"build": build_with_startup_hook, "build_startup_hook": build_startup_hook})
