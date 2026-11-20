from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector


def __getattr__(name):
    # `visualize_setup` pulls in Plotly and the 1700-line plot module. Neither is needed to
    # simulate, so they are loaded on first use instead of on every `import differometor`.
    if name == "visualize_setup":
        from differometor.plot import visualize_setup

        return visualize_setup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
