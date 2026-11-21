"""
Interferometer setup builder and converter utilities.

This module defines:

- A small graph-like container (Setup) with:
  - nodes (optical and mechanical components)
  - edges (spaces connecting components)
  - a parameter list that can be used for optimization workflows

- Several predefined setups:
  - voyager
  - aligo
  - uifo (quasi-universal interferometer grid)

- Utilities:
  - constrain_inter_grid_cell_spaces (parameter tying for UIFO grids)
  - differometor_to_finesse (export to Finesse text format)

Design notes
------------
- Node names are not allowed to contain underscores. Underscores are reserved for
  edge identifiers of the form "source_target".
- Edges are stored internally with tuple keys (source, target) and are presented
  externally as strings "source_target" for convenience in some APIs.
- The "parameters" list stores (component_identifier, property_name) pairs.
  - For nodes, component_identifier is the node name.
  - For edges (spaces), component_identifier is "source_target".

- The `not_optimizable` argument can be:
  - None: treat as empty list (everything is optimizable by default)
  - True: mark the entire component as not optimizable
  - list: property-level exclusions. Each item must be a (name, property_name) pair.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from differometor.components import DEFAULT_PROPERTIES


# Public types used throughout this module
NodeName = str
EdgeKey = Tuple[NodeName, NodeName]          # internal representation
EdgeName = str                               # external "source_target" representation
PropertyName = str

# (component_identifier, property_name)
ParameterPair = Tuple[str, str]

NotOptimizable = Union[None, bool, List[ParameterPair]]


class Nodes:
    """
    Lightweight view wrapper over the internal node dictionary.

    Instances of this class are created and maintained by Setup. It provides:
    - iteration in a style similar to networkx: `for name, data in setup.nodes(data=True): ...`
    - `__getitem__` for direct lookup of a node data dictionary

    Parameters
    ----------
    nodes:
        The underlying node dictionary owned by Setup. Keys are node names and
        values are node metadata dictionaries.
    """

    def __init__(self, nodes: Dict[NodeName, Dict[str, Any]]):
        self._nodes = nodes

    def __iter__(self) -> Iterator[Tuple[NodeName, Dict[str, Any]]]:
        """Iterate over (node_name, node_data)."""
        return iter(self._nodes.items())

    def __call__(self, data: bool = True) -> Iterator[Any]:
        """
        Mimic a subset of the networkx `.nodes(data=...)` behavior.

        Parameters
        ----------
        data:
            If True, yield (node_name, node_data).
            If False, yield node_name.

        Returns
        -------
        iterator:
            Iterator over nodes according to `data`.
        """
        if data:
            return iter(self._nodes.items())
        return iter(self._nodes.keys())

    def __getitem__(self, node: NodeName) -> Dict[str, Any]:
        """
        Retrieve metadata for a given node.

        Raises
        ------
        KeyError:
            If `node` is not present.
        """
        if node in self._nodes:
            return self._nodes[node]
        raise KeyError(f"Node '{node}' not found in the setup.")


class Edges:
    """
    Lightweight view wrapper over the internal edge dictionary.

    Internally, edges are stored as keys `(source, target)`.

    Externally, some interfaces address edges by the string "source_target".
    This class supports both:
    - iteration similar to networkx: `for src, tgt, data in setup.edges(data=True): ...`
    - `__getitem__` with the string edge name "source_target"

    Parameters
    ----------
    edges:
        The underlying edge dictionary owned by Setup. Keys are (source, target)
        tuples and values are edge metadata dictionaries.
    """

    def __init__(self, edges: Dict[EdgeKey, Dict[str, Any]]):
        self._edges = edges

    def __iter__(self) -> Iterator[Tuple[NodeName, NodeName, Dict[str, Any]]]:
        """Iterate over (source, target, edge_data)."""
        return iter((src, tgt, data) for (src, tgt), data in self._edges.items())

    def __getitem__(self, edge: EdgeName) -> Dict[str, Any]:
        """
        Retrieve metadata for a given edge addressed as "source_target".

        Parameters
        ----------
        edge:
            String edge name in the format "source_target".

        Raises
        ------
        KeyError:
            If the edge is not present.
        ValueError:
            If `edge` does not contain a single underscore separator.
        """
        if "_" not in edge:
            raise ValueError(
                f"Edge '{edge}' must be addressed as 'source_target' (one underscore separator)."
            )
        source, target = edge.split("_", 1)
        if (source, target) in self._edges:
            return self._edges[(source, target)]
        raise KeyError(f"Edge '{edge}' not found in the setup.")

    def __call__(self, data: bool = True) -> Iterator[Any]:
        """
        Mimic a subset of the networkx `.edges(data=...)` behavior.

        Parameters
        ----------
        data:
            If True, yield (source, target, edge_data).
            If False, yield (source, target).

        Returns
        -------
        iterator:
            Iterator over edges according to `data`.
        """
        if data:
            return iter((src, tgt, d) for (src, tgt), d in self._edges.items())
        return iter((src, tgt) for (src, tgt) in self._edges)


class Setup:
    """
    Container describing an optical (and optomechanical) interferometer-like setup.

    The setup consists of:
    - nodes: components such as laser, mirror, beamsplitter, squeezer, detector, free_mass, signal
    - edges: spaces connecting nodes with length and refractive index properties
    - parameters: a flat list of (component_identifier, property_name) pairs that represent
      what can be optimized (unless excluded)

    Attributes
    ----------
    parameters:
        List of (component_identifier, property_name) pairs to optimize.
        For nodes, component_identifier is the node name.
        For spaces, component_identifier is the string "source_target".
    nodes:
        A Nodes view over the internal node dictionary.
    edges:
        An Edges view over the internal edge dictionary.
    default_properties:
        DEFAULT_PROPERTIES imported from differometor.components
    """

    def __init__(self):
        self.parameters: List[ParameterPair] = []
        self._nodes: Dict[NodeName, Dict[str, Any]] = {}
        self._edges: Dict[EdgeKey, Dict[str, Any]] = {}

        # Lightweight views that provide convenient iteration and lookup
        self.nodes = Nodes(self._nodes)
        self.edges = Edges(self._edges)

        # Property templates per component type
        self.default_properties = DEFAULT_PROPERTIES

    def add(
        self,
        component: str,
        name: str,
        not_optimizable: NotOptimizable = None,
        target: Optional[str] = None,
        port: Optional[str] = None,
        direction: Optional[str] = None,
        auxiliary: Optional[bool] = None,
        detector1: Optional[str] = None,
        detector2: Optional[str] = None,
        **properties: Any,
    ) -> None:
        """
        Add a node (component) to the setup.

        Parameters
        ----------
        component:
            Component type name. Must exist in DEFAULT_PROPERTIES.
            Common values include: "laser", "mirror", "beamsplitter", "squeezer",
            "frequency", "signal", "free_mass", "detector", "qnoised", "qhd",
            "directional_beamsplitter", "nothing".
        name:
            Node name identifier. Must not contain underscores. Underscores are reserved
            for edge identifiers "source_target".
        not_optimizable:
            Controls how this node contributes to `self.parameters`.

            Accepted values:
            - None: treat as an empty list, meaning all properties are optimizable
              (subject to special rules for certain components).
            - True: exclude this entire node from the parameter list.
            - list of (node_name, property_name): exclude individual properties.

            Special rule:
            - For "signal" and "frequency" components, parameters are not added by default.
        target:
            Optional target reference, used by certain components such as "free_mass",
            "signal", "detector", and "qnoised". Accepted formats:
            - node name, for example "bs"
            - edge name "source_target", for example "l0_prm"
            - node property "node_amplitude" or "node_frequency" for signal components
        port:
            Optional port name. Must be one of: "left", "top", "right", "bottom".
        direction:
            Optional direction string. Must be one of: "in", "out".
        auxiliary:
            Optional auxiliary flag. Must be True or False if provided.
        detector1, detector2:
            Optional detector node names used for "qhd" components.

        **properties:
            Component-specific properties, merged with DEFAULT_PROPERTIES[component].

            For "mirror" and "beamsplitter", an additional convenience is supported:
            - You may pass transmissivity instead of reflectivity.
              In that case, reflectivity is derived and normalized with respect to loss.

        Raises
        ------
        ValueError:
            For invalid component names, invalid property sets, invalid ports or directions,
            invalid targets, or underscore usage in node names.
        """
        from differometor._setup_bau import add

        return add(self, component, name, not_optimizable, target, port, direction, auxiliary, detector1, detector2, **properties)

    def space(
        self,
        source: str,
        target: str,
        not_optimizable: NotOptimizable = None,
        source_port: str = "right",
        target_port: str = "left",
        **properties: Any,
    ) -> None:
        """
        Add a space (edge) between two existing nodes.

        Parameters
        ----------
        source:
            Source node name. Must exist in the setup.
        target:
            Target node name. Must exist in the setup.
        not_optimizable:
            Controls how this space contributes to `self.parameters`.

            Accepted values:
            - None: treat as empty list knowing all space properties are optimizable
            - True: exclude this entire space from parameter list
            - list of (edge_name, property_name) pairs to exclude individually

            For spaces, the component identifier in the parameter list is "source_target".
        source_port:
            Which port on the source node the space starts at. Default is "right".
        target_port:
            Which port on the target node the space ends at. Default is "left".
        **properties:
            Space properties, merged with DEFAULT_PROPERTIES["space"].
            The default properties usually include length and refractive_index.

        Raises
        ------
        ValueError:
            If source or target nodes do not exist, or if properties are invalid.
        """
        from differometor._setup_bau import space

        return space(self, source, target, not_optimizable, source_port, target_port, **properties)

    def to_data(self) -> dict:
        """
        Serialize this setup into a plain Python dictionary.

        This is intended for:
        - saving to JSON (after ensuring all values are JSON-serializable)
        - transferring setups between processes

        Returns
        -------
        data:
            Dictionary with keys:
            - "parameters": list
            - "nodes": dict
            - "edges": dict with string keys "source_target"
        """
        edges_serializable = {
            f"{src}_{tgt}": data for (src, tgt), data in self._edges.items()
        }
        return {
            "parameters": list(self.parameters),
            "nodes": dict(self._nodes),
            "edges": edges_serializable,
        }

    @classmethod
    def from_data(cls, data: dict) -> "Setup":
        """
        Rebuild a Setup instance from the dictionary created by `to_data`.

        Parameters
        ----------
        data:
            Dictionary containing "parameters", "nodes", and "edges".

        Returns
        -------
        setup:
            A reconstructed Setup instance with Nodes and Edges view wrappers attached.

        Notes
        -----
        - Edge keys are expected as strings "source_target".
        - Node names may not contain underscores, so a split on the first underscore is safe.
        """
        setup = cls()
        setup.parameters = list(data["parameters"])
        setup._nodes = dict(data["nodes"])
        setup._edges = {
            tuple(edge_str.split("_", 1)): edge_data
            for edge_str, edge_data in data["edges"].items()
        }

        # Re-link view wrappers after replacing the underlying dictionaries
        setup.nodes = Nodes(setup._nodes)
        setup.edges = Edges(setup._edges)
        return setup


# ----------------------------------------------------------------------
# Predefined setups
# ----------------------------------------------------------------------

# voyager, aligo and uifo live in differometor/_beispiel_setups.py and are loaded on
# first access, so a plain import of this module does not compile them.
_EXAMPLE_SETUPS = ("voyager", "aligo", "uifo")


def __getattr__(name):
    if name in _EXAMPLE_SETUPS:
        from differometor import _beispiel_setups

        function = getattr(_beispiel_setups, name)
        globals()[name] = function
        return function
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXAMPLE_SETUPS))


# ----------------------------------------------------------------------
# Optimization constraint utilities
# ----------------------------------------------------------------------

def constrain_inter_grid_cell_spaces(
    component_property_pairs: List[ParameterPair],
    optimized_properties: List[str],
) -> list:
    """
    UIFO-specific parameter constraint helper.

    This function filters and groups parameters so that parallel spaces in different
    grid cells share the same optimization variable. This preserves the grid structure
    when optimizing lengths.

    The input `component_property_pairs` is typically `setup.parameters`.

    Behavior
    --------
    - First, parameters are filtered so only those whose property name is in `optimized_properties`
      remain.
    - Parameters with property "length" are treated specially:
      - Some interior connections are grouped:
        - Horizontal inter-cell connections (mr*_ml*) are grouped by a column index.
        - Vertical inter-cell connections (mt*_mb*) are grouped by a row index.
      - Spaces that involve "center" or "boundary" are excluded from length grouping.
    - Non-length parameters are left as individual entries.

    Parameters
    ----------
    component_property_pairs:
        List of (component_identifier, property_name) pairs.
    optimized_properties:
        List of property names that are actually optimized, for example ["length"].

    Returns
    -------
    constrained_pairs:
        A list containing a mix of:
        - single [component_identifier, property_name]
        - grouped lists of [component_identifier, property_name] entries

        If a group has only one element, it is flattened back to a single entry.

    Notes
    -----
    - This grouping logic assumes UIFO naming conventions:
      - mirrors named mlXY, mrXY, mtXY, mbXY
      - inter-cell edges named like "mr{x}{y-1}_ml{x}{y}" and "mt{x}{y}_mb{x-1}{y}"
    """
    # Keep only properties that are actually optimized
    component_property_pairs = [
        [component_name, property_name]
        for component_name, property_name in component_property_pairs
        if property_name in optimized_properties
    ]

    constrained_pairs: List[Any] = []

    # Two group dictionaries:
    # - index 0: horizontal spaces
    # - index 1: vertical spaces
    constrained_pair_dicts = [defaultdict(list), defaultdict(list)]

    for component_name, property_name in component_property_pairs:
        if property_name == "length":
            # Skip spaces that connect to centers or boundaries, keep them independent or ignore
            if "center" in component_name or "boundary" in component_name:
                continue

            # Horizontal inter-cell spaces: "...mr..._ml..."
            if "mr" in component_name and "_ml" in component_name:
                # Group key is derived from a coordinate character in the naming convention
                constrained_pair_dicts[0][component_name.split("_")[0][-1]].append([component_name, property_name])

            # Vertical inter-cell spaces: "...mt..._mb..."
            elif "mt" in component_name and "_mb" in component_name:
                constrained_pair_dicts[1][component_name.split("_")[0][-2]].append([component_name, property_name])

            else:
                constrained_pairs.append([component_name, property_name])

        else:
            constrained_pairs.append([component_name, property_name])

    # Append grouped constraints
    for constrained_pair_dict in constrained_pair_dicts:
        constrained_pairs.extend([constrained_pair_dict[key] for key in constrained_pair_dict])

    # Remove empty entries and flatten one-element groups
    constrained_pairs = [parameter for parameter in constrained_pairs if parameter]
    constrained_pairs = [parameter[0] if len(parameter) == 1 else parameter for parameter in constrained_pairs]
    return constrained_pairs
