"""
Predefined example setups (voyager, aligo, uifo).

Split out of `differometor.setups` so that importing the package does not have to
compile them. They remain importable as `differometor.setups.voyager` etc.; the
setups module loads this file on first access.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from differometor.setups import Setup


def voyager(mode: str = "space_modulation") -> tuple[Setup, list]:
    """
    Construct and return the predefined Voyager setup.

    Parameters
    ----------
    mode:
        Modulation type. Supported values:
        - "space_modulation"
        - "amplitude_modulation"
        - "frequency_modulation"

    Returns
    -------
    setup:
        The constructed setup instance.
    parameters:
        The list of optimizable parameters derived from the setup.

    Raises
    ------
    ValueError:
        If `mode` is not one of the supported values.
    """
    S = Setup()
    S.add("laser", "l0", power=153, phase=0)
    S.add("mirror", "prm", transmissivity=0.049, loss=5e-06, tuning=0)
    S.add("beamsplitter", "bs", transmissivity=0.5, loss=5e-06, tuning=63.63961030678928, alpha=45)
    S.add("mirror", "itmy", transmissivity=0.002, loss=5e-06, tuning=0)
    S.add("mirror", "etmy", transmissivity=1.5e-05, loss=5e-06, tuning=0)
    S.add("mirror", "itmx", transmissivity=0.002, loss=5e-06, tuning=0)
    S.add("mirror", "etmx", transmissivity=1.5e-05, loss=5e-06, tuning=0)
    S.add("mirror", "srm", transmissivity=0.046, loss=5e-06, tuning=90)
    S.add("directional_beamsplitter", "dbs1")
    S.add("directional_beamsplitter", "dbs2")
    S.add("squeezer", "sq", db=10, angle=0)
    S.add("mirror", "fm1", transmissivity=0.1e-2, loss=5e-06, tuning=0)
    S.add("mirror", "fm2", transmissivity=1.5e-05, loss=5e-06, tuning=-0.014)
    S.add("beamsplitter", "bhbs", transmissivity=0.5, loss=5e-06, tuning=1e-07, alpha=45)
    S.add("laser", "lo", power=0.01, phase=0)

    # Mechanical degrees of freedom (free-mass suspensions) targeting optical nodes
    S.add("free_mass", "prmsus", mass=29.243802983873618, target="prm")
    S.add("free_mass", "bssus", mass=48.634040943805395, target="bs")
    S.add("free_mass", "itmysus", mass=200, target="itmy")
    S.add("free_mass", "etmysus", mass=200, target="etmy")
    S.add("free_mass", "itmxsus", mass=200, target="itmx")
    S.add("free_mass", "etmxsus", mass=200, target="etmx")
    S.add("free_mass", "srmsus", mass=50, target="srm")

    # Optical connections (spaces)
    S.space("l0", "prm", length=1)
    S.space("prm", "bs", length=1)
    S.space("bs", "itmy", length=1, source_port="top")
    S.space("itmy", "etmy", length=4000)
    S.space("bs", "itmx", length=1, source_port="right")
    S.space("itmx", "etmx", length=4000)
    S.space("bs", "srm", length=10, source_port="bottom")
    S.space("srm", "dbs1", length=1, target_port="left")
    S.space("sq", "dbs2", length=1, target_port="top")
    S.space("dbs1", "dbs2", length=10, source_port="top", target_port="right")
    S.space("dbs2", "fm1", length=1, source_port="left")
    S.space("fm1", "fm2", length=300)
    S.space("dbs1", "bhbs", length=1, source_port="right", target_port="left")
    S.space("lo", "bhbs", length=10, target_port="bottom")

    # Frequency node used by signal generation in some workflows
    S.add("frequency", "f", frequency=1)

    # Signal definitions depend on the chosen modulation mode
    if mode == "space_modulation":
        # Signals target specific spaces (edges) by name "source_target"
        S.add("signal", "fl0prm", target="l0_prm")
        S.add("signal", "fprmbs", target="prm_bs")
        S.add("signal", "fbsitmy", target="bs_itmy", phase=180)
        S.add("signal", "fitmyetmy", target="itmy_etmy", phase=180)
        S.add("signal", "fbsitmx", target="bs_itmx")
        S.add("signal", "fitmxetmx", target="itmx_etmx")
        S.add("signal", "bssrm", target="bs_srm", phase=180)
    elif mode == "amplitude_modulation":
        # Signals target node properties by "node_amplitude"
        S.add("signal", "fl0", target="l0_amplitude", amplitude=("l0_power", jnp.sqrt))
        S.add("signal", "flo", target="lo_amplitude", amplitude=("lo_power", jnp.sqrt))
    elif mode == "frequency_modulation":
        # Signals target node properties by "node_frequency"
        S.add("signal", "fl0", target="l0_frequency")
        S.add("signal", "flo", target="lo_frequency")
    else:
        raise ValueError(
            "Invalid mode. Choose from 'space_modulation', 'amplitude_modulation', or 'frequency_modulation'."
        )

    # Balanced homodyne readout noise and detectors
    S.add("qnoised", "noise-top", target="bhbs", port="top", direction="out", auxiliary=True)
    S.add("qnoised", "noise-right", target="bhbs", port="right", direction="out", auxiliary=True)
    S.add("qhd", "noise", detector1="noise-top", detector2="noise-right", phase=180)
    S.add("detector", "detector-top", target="bhbs", port="top", direction="out")
    S.add("detector", "detector-right", target="bhbs", port="right", direction="out")

    return S, S.parameters


def aligo(mode: str = "space_modulation") -> tuple[Setup, list]:
    """
    Construct and return a simplified aLIGO setup with optomechanics and squeezing.

    Parameters
    ----------
    mode:
        Modulation type. Supported values:
        - "space_modulation"
        - "amplitude_modulation"
        - "frequency_modulation"

    Returns
    -------
    setup:
        The constructed setup instance.
    parameters:
        The list of optimizable parameters derived from the setup.

    Raises
    ------
    ValueError:
        If `mode` is not one of the supported values.
    """
    Larm = 3995
    itmT = 0.014
    mirrorL = 37.5e-6
    etmT = 5e-6
    Mtm = 40

    S = Setup()
    S.add("laser", "L0", power=125)
    S.add("beamsplitter", "bs", reflectivity=0.5, loss=0, alpha=45)
    S.add("mirror", "prm", transmissivity=0.03, loss=mirrorL, tuning=90)
    S.add("mirror", "itmx", transmissivity=itmT, loss=mirrorL, tuning=90)
    S.add("mirror", "etmx", transmissivity=etmT, loss=mirrorL, tuning=89.999875)
    S.add("mirror", "itmy", transmissivity=itmT, loss=mirrorL, tuning=0)
    S.add("mirror", "etmy", transmissivity=etmT, loss=mirrorL, tuning=0.000125)
    S.add("mirror", "srm", transmissivity=0.2, loss=mirrorL, tuning=-90)
    S.add("squeezer", "sq1", db=10, angle=90)

    # Mechanical degrees of freedom for the test masses
    S.add("free_mass", "itmxsus", mass=Mtm, target="itmx")
    S.add("free_mass", "etmxsus", mass=Mtm, target="etmx")
    S.add("free_mass", "itmysus", mass=Mtm, target="itmy")
    S.add("free_mass", "etmysus", mass=Mtm, target="etmy")

    # Optical connections
    S.space("L0", "prm")
    S.space("prm", "bs", length=53)
    S.space("bs", "itmx", length=4.5)
    S.space("itmx", "etmx", length=Larm)
    S.space("bs", "itmy", length=4.45, source_port="top")
    S.space("itmy", "etmy", length=Larm)
    S.space("bs", "srm", length=50.525, source_port="bottom")
    S.space("sq1", "srm", target_port="right")

    # Signal frequency
    S.add("frequency", "f", frequency=5)

    # Mode-dependent signal injection
    if mode == "space_modulation":
        S.add("signal", "darmx", target="itmx_etmx")
        S.add("signal", "darmy", target="itmy_etmy", phase=180)
    elif mode == "frequency_modulation":
        S.add("signal", "fL0", target="L0_frequency")
    elif mode == "amplitude_modulation":
        S.add("signal", "fL0", target="L0_amplitude", amplitude=("L0_power", jnp.sqrt))
    else:
        raise ValueError(
            "Invalid mode. Choose from 'space_modulation', 'amplitude_modulation', or 'frequency_modulation'."
        )

    # Readout noise and detector at the signal recycling mirror output port
    S.add("qnoised", "noise", target="srm", port="right", direction="out")
    S.add("detector", "detector", target="srm", port="right", direction="out")

    return S, S.parameters


def uifo(
    size: int,
    centers: Optional[dict] = None,
    boundaries: Optional[dict] = None,
    random: bool = False,
    mode: str = "space_modulation",
    verbose: bool = False,
    random_seed: Optional[int] = None,
) -> tuple[Setup, list]:
    """
    Define a quasi-universal interferometer (UIFO) as a grid of unit cells.

    The UIFO is constructed as:
    - An interior grid of unit cells (size x size).
    - Four boundaries around the grid, each boundary cell contains a source or readout.

    Parameters
    ----------
    size:
        Grid size. For example, size=3 creates a 3x3 interior grid.
    centers:
        Dictionary defining the centers of the unit cells.

        Keys are "xy" coordinates (as strings), values are tuples:
        (component_type, left_port_position)

        - component_type is one of:
          ["beamsplitter", "directional_beamsplitter"]
        - left_port_position is one of:
          ["left", "top", "right", "bottom"]

        Unspecified centers are filled with defaults or random choices (if random=True).
    boundaries:
        Dictionary defining boundary sources or readouts.

        Keys are "xy" coordinates on the boundary ring, values are strings:
        - "laser"
        - "squeezer"
        - "detector"
        - "balanced_homodyne"

        Unspecified boundaries are filled with defaults or random choices (if random=True).

        If random=True and neither "detector" nor "balanced_homodyne" is present, at least one
        is inserted at a random boundary position.
    random:
        If True, choose random center and boundary configurations (with a detector guaranteed).
    mode:
        Modulation mode:
        - "space_modulation"
        - "frequency_modulation"
        - "amplitude_modulation"
    verbose:
        If True, return the resolved centers and boundaries dictionaries as well.
    random_seed:
        Seed for reproducible random configuration. If None, a fresh seed is used.

    Returns
    -------
    setup:
        The setup object containing all components and their connections.
    component_property_pairs:
        The list of optimizable parameters of the setup.
    centers (only if verbose=True):
        The fully resolved centers mapping.
    boundaries (only if verbose=True):
        The fully resolved boundaries mapping.

    Raises
    ------
    ValueError:
        If `mode` is invalid.
    """
    rng = np.random.default_rng(random_seed)

    # Resolve center defaults
    if random:
        orientations = ["left", "top", "right", "bottom"]
        center_choices = ["beamsplitter", "directional_beamsplitter"]
        default_center_function: Callable[[], Tuple[str, str]] = lambda: (
            rng.choice(center_choices),
            rng.choice(orientations),
        )
    else:
        default_center_function = lambda: ("beamsplitter", "left")

    default_centers = defaultdict(default_center_function)
    default_centers.update(centers or {})
    centers = default_centers

    # Resolve boundary defaults
    if random:
        boundary_choices = ["laser", "squeezer"]
        default_boundary_function: Callable[[], str] = lambda: rng.choice(boundary_choices)
    else:
        default_boundary_function = lambda: "laser"

    default_boundaries = defaultdict(default_boundary_function)
    default_boundaries.update(boundaries or {})
    boundaries = default_boundaries

    # Ensure at least one detector or balanced homodyne readout if random=True
    if random:
        if "detector" not in default_boundaries.values() and "balanced_homodyne" not in default_boundaries.values():
            side = rng.choice([0, size + 1])
            position = rng.choice(range(1, size + 1))
            detector_type = rng.choice(["detector", "balanced_homodyne"])
            if rng.choice([True, False]):
                default_boundaries[f"{side}{position}"] = detector_type
            else:
                default_boundaries[f"{position}{side}"] = detector_type

    def unit_cell(
        S: Setup,
        x: int,
        y: int,
        center: str = "beamsplitter",
        left_port_position: str = "left",
    ) -> None:
        """
        Add a single interior unit cell at grid coordinate (x, y).

        The unit cell contains:
        - a central beamsplitter-like element (beamsplitter or directional_beamsplitter)
        - four surrounding mirrors (left, right, top, bottom)
        - free_mass suspension nodes for each mirror and for the central beamsplitter if present
        - spaces connecting the center to each mirror

        Signal nodes (only for space_modulation) are attached to each center-to-mirror space,
        with phases determined by physical direction:
        - horizontal spaces: phase 0
        - vertical spaces: phase 180
        """
        if center == "beamsplitter":
            S.add("beamsplitter", f"center{x}{y}")
            S.add("free_mass", f"center{x}{y}sus", target=f"center{x}{y}")
        elif center == "directional_beamsplitter":
            S.add("directional_beamsplitter", f"center{x}{y}")

        # Mirrors around the center: ml, mr, mt, mb
        S.add("mirror", f"ml{x}{y}")
        S.add("mirror", f"mr{x}{y}")
        S.add("mirror", f"mt{x}{y}")
        S.add("mirror", f"mb{x}{y}")

        # Free masses for each mirror
        S.add("free_mass", f"ml{x}{y}sus", target=f"ml{x}{y}")
        S.add("free_mass", f"mr{x}{y}sus", target=f"mr{x}{y}")
        S.add("free_mass", f"mt{x}{y}sus", target=f"mt{x}{y}")
        S.add("free_mass", f"mb{x}{y}sus", target=f"mb{x}{y}")

        # Orientation mapping: which beamsplitter ports correspond to which physical direction
        ports = {
            "left": ["left", "top", "right", "bottom"],
            "top": ["bottom", "left", "top", "right"],
            "right": ["right", "bottom", "left", "top"],
            "bottom": ["top", "right", "bottom", "left"],
        }

        mirrors = ["ml", "mt", "mr", "mb"]
        for i, port in enumerate(ports[left_port_position]):
            # Centers connect outward to mirrors, always using the mirror left port by default.
            S.space(f"center{x}{y}", f"{mirrors[i]}{x}{y}", length=1, source_port=port)

        if mode == "space_modulation":
            # Signals target edge names "centerxy_m?xy"
            S.add("signal", f"scenter{x}{y}ml{x}{y}", target=f"center{x}{y}_ml{x}{y}", phase=0)
            S.add("signal", f"scenter{x}{y}mr{x}{y}", target=f"center{x}{y}_mr{x}{y}", phase=0)
            S.add("signal", f"scenter{x}{y}mt{x}{y}", target=f"center{x}{y}_mt{x}{y}", phase=180)
            S.add("signal", f"scenter{x}{y}mb{x}{y}", target=f"center{x}{y}_mb{x}{y}", phase=180)

    def boundary_cell(
        S: Setup,
        x: int,
        y: int,
        boundary: str = "laser",
        mass: bool = True,
        position: str = "left",
    ) -> None:
        """
        Add a boundary cell at coordinate (x, y) outside the main interior grid.

        The boundary cell always includes:
        - a mirror node mxy
        - optionally a free_mass suspension for that mirror

        Depending on `boundary`, it also includes:
        - "detector": a detector and a qnoised readout at the mirror left port
        - "balanced_homodyne": a local oscillator laser, a beamsplitter, readout noise nodes, and detectors
        - "laser" or "squeezer": a source connected to the mirror

        Parameters
        ----------
        position:
            Indicates which side of the overall grid this boundary cell is on:
            "left", "right", "top", "bottom". Used to set signal phases in space_modulation mode.
        """
        S.add("mirror", f"m{x}{y}")

        if mass:
            S.add("free_mass", f"m{x}{y}sus", target=f"m{x}{y}")

        # Sources always connect to the mirror left port
        if boundary == "detector":
            S.add("detector", f"boundary{x}{y}detector", target=f"m{x}{y}", port="left", direction="out")
            S.add("qnoised", f"boundary{x}{y}noise", target=f"m{x}{y}", port="left", direction="out")

        elif boundary == "balanced_homodyne":
            # Local oscillator is not optimizable by default
            S.add("laser", f"boundary{x}{y}lo", power=0.01, phase=0, not_optimizable=True)
            S.add("beamsplitter", f"boundary{x}{y}bhbs")
            S.add("qnoised", f"boundary{x}{y}noise-top", target=f"boundary{x}{y}bhbs",
                  port="top", direction="out", auxiliary=True)
            S.add("qnoised", f"boundary{x}{y}noise-right", target=f"boundary{x}{y}bhbs",
                  port="right", direction="out", auxiliary=True)
            S.add("qhd", f"boundary{x}{y}noise", detector1=f"boundary{x}{y}noise-top",
                  detector2=f"boundary{x}{y}noise-right")
            S.add("detector", "detector-top", target=f"boundary{x}{y}bhbs", port="top", direction="out")
            S.add("detector", "detector-right", target=f"boundary{x}{y}bhbs", port="right", direction="out")

            # Connect local oscillator and mirror into the balanced homodyne beamsplitter
            S.space(f"boundary{x}{y}lo", f"boundary{x}{y}bhbs", target_port="bottom")
            S.space(f"m{x}{y}", f"boundary{x}{y}bhbs", source_port="left", target_port="left")

            if mode == "space_modulation":
                # Space signal phases depend on whether the space is vertical or horizontal
                S.add(
                    "signal",
                    f"sm{x}{y}boundary{x}{y}bhbs",
                    target=f"m{x}{y}_boundary{x}{y}bhbs",
                    phase=180 if position in ["top", "bottom"] else 0,
                )
                S.add(
                    "signal",
                    f"sboundary{x}{y}loboundary{x}{y}bhbs",
                    target=f"boundary{x}{y}lo_boundary{x}{y}bhbs",
                    phase=180 if position in ["left", "right"] else 0,
                )
            if mode == "amplitude_modulation":
                S.add(
                    "signal",
                    f"sboundary{x}{y}lo",
                    target=f"boundary{x}{y}lo_amplitude",
                    amplitude=(f"boundary{x}{y}lo_power", jnp.sqrt),
                )
            if mode == "frequency_modulation":
                S.add("signal", f"sboundary{x}{y}lo", target=f"boundary{x}{y}lo_frequency")

        elif boundary in ["laser", "squeezer"]:
            S.add(boundary, f"boundary{x}{y}")
            S.space(f"boundary{x}{y}", f"m{x}{y}")

            if mode == "space_modulation":
                # For vertical boundary connections (top and bottom), use phase 180
                S.add(
                    "signal",
                    f"sboundary{x}{y}m{x}{y}",
                    target=f"boundary{x}{y}_m{x}{y}",
                    phase=180 if position in ["top", "bottom"] else 0,
                )

            if boundary == "laser" and mode == "amplitude_modulation":
                S.add(
                    "signal",
                    f"sboundary{x}{y}",
                    target=f"boundary{x}{y}_amplitude",
                    amplitude=(f"boundary{x}{y}_power", jnp.sqrt),
                )
            if boundary == "laser" and mode == "frequency_modulation":
                S.add("signal", f"sboundary{x}{y}", target=f"boundary{x}{y}_frequency")

        else:
            raise ValueError(
                f"Invalid boundary component '{boundary}'. "
                "Choose from 'laser', 'squeezer', 'detector', 'balanced_homodyne'."
            )

    def cell_grid(S: Setup, n: int) -> None:
        """
        Build the interior grid of unit cells and their interconnections.

        Steps
        -----
        1) Create each unit cell and its internal connections.
        2) Create connections between neighboring cells:
           - Vertical: mt(x,y) connects to mb(x-1,y) for x > 1
           - Horizontal: mr(x,y-1) connects to ml(x,y) for y > 1
        """
        # Add unit cells
        for x in range(1, n + 1):
            for y in range(1, n + 1):
                center, left_port_position = centers[f"{x}{y}"]
                unit_cell(S, x, y, center=center, left_port_position=left_port_position)

        # Connect unit cells to neighbors inside the grid
        for x in range(1, n + 1):
            for y in range(1, n + 1):
                if x > 1:
                    # Use right ports because left ports are taken by center connections
                    S.space(f"mt{x}{y}", f"mb{x-1}{y}", source_port="right", target_port="right")
                    if mode == "space_modulation":
                        S.add("signal", f"smt{x}{y}mb{x-1}{y}", target=f"mt{x}{y}_mb{x-1}{y}", phase=180)
                if y > 1:
                    S.space(f"mr{x}{y-1}", f"ml{x}{y}", source_port="right", target_port="right")
                    if mode == "space_modulation":
                        S.add("signal", f"smr{x}{y-1}ml{x}{y}", target=f"mr{x}{y-1}_ml{x}{y}")

    S = Setup()
    S.add("frequency", "f")
    cell_grid(S, size)

    # Left and right boundaries
    for x in range(1, size + 1):
        boundary_cell(S, x, 0, boundary=boundaries[f"{x}0"], position="left")
        S.space(f"m{x}0", f"ml{x}1", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"sm{x}0ml{x}1", target=f"m{x}0_ml{x}1")

        boundary_cell(S, x, size + 1, boundary=boundaries[f"{x}{size+1}"], position="right")
        S.space(f"mr{x}{size}", f"m{x}{size+1}", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"smr{x}{size}m{x}{size+1}", target=f"mr{x}{size}_m{x}{size+1}")

    # Top and bottom boundaries
    for y in range(1, size + 1):
        boundary_cell(S, 0, y, boundary=boundaries[f"0{y}"], position="top")
        S.space(f"m0{y}", f"mt1{y}", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"sm0{y}mt1{y}", target=f"m0{y}_mt1{y}", phase=180)

        boundary_cell(S, size + 1, y, boundary=boundaries[f"{size+1}{y}"], position="bottom")
        S.space(f"mb{size}{y}", f"m{size+1}{y}", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"smb{size}{y}m{size+1}{y}", target=f"mb{size}{y}_m{size+1}{y}", phase=180)

    if verbose:
        # Convert defaultdict values to plain serializable strings
        for key in default_centers:
            default_centers[key] = (str(default_centers[key][0]), str(default_centers[key][1]))
        for key in default_boundaries:
            default_boundaries[key] = str(default_boundaries[key])
        return S, S.parameters, default_centers, default_boundaries

    return S, S.parameters
