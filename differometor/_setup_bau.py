"""
Bodies of `Setup.add` and `Setup.space`.

Split out of `differometor.setups` so that importing the package (and loading a setup
with `Setup.from_data`) does not have to compile them. `Setup.add` and `Setup.space`
keep their signatures and docstrings and call these functions on first use.
"""

from __future__ import annotations

from typing import Any, Optional

from differometor.components import DEFAULT_PROPERTIES
from differometor.setups import NotOptimizable


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
    if "_" in name:
        raise ValueError(
            f"Node name '{name}' cannot contain underscores. Use '-' instead."
        )

    # Normalize not_optimizable
    if not_optimizable is None:
        not_optimizable = []

    # Convenience: allow transmissivity for mirror and beamsplitter
    if component in ["mirror", "beamsplitter"]:
        if "reflectivity" in properties and "transmissivity" in properties:
            raise ValueError(
                "Cannot specify both 'reflectivity' and 'transmissivity'. "
                "Use 'reflectivity' and 'loss' or 'transmissivity' and 'loss' instead."
            )

        if "transmissivity" in properties:
            # Merge defaults first so loss is known
            properties = {**self.default_properties[component], **properties}
            transmissivity = properties.pop("transmissivity")

            # Convert transmissivity to normalized reflectivity with respect to loss.
            # Internally, reflectivity is stored as a factor on (1 - loss).
            properties["reflectivity"] = (1 - transmissivity - properties["loss"]) / (
                1 - properties["loss"]
            )
        elif "reflectivity" in properties:
            # Merge defaults first so loss is known
            properties = {**self.default_properties[component], **properties}

            # Normalize reflectivity by (1 - loss) to store it in the internal convention.
            properties["reflectivity"] = properties["reflectivity"] / (
                1 - properties["loss"]
            )

    # Merge defaults for every component
    try:
        properties = {**self.default_properties[component], **properties}
    except KeyError as exc:
        raise ValueError(f"Component '{component}' is not recognized.") from exc

    # Validate that no unknown properties were provided
    if len(properties) != len(self.default_properties[component]):
        raise ValueError(
            f"Component '{component}' has the properties "
            f"{list(self.default_properties[component].keys())} "
            f"but received {list(properties.keys())}."
        )

    # Add optimizable parameters for regular components
    if component not in ["signal", "frequency"]:
        if not_optimizable is not True:
            self.parameters.extend(
                [
                    (name, property_name)
                    for property_name in properties.keys()
                    if (name, property_name) not in not_optimizable  # type: ignore[operator]
                ]
            )

    # Store node
    self._nodes[name] = {"component": component, "properties": properties}

    # Parse and validate target references (if any)
    if target is not None:
        # Target might be edge-like "src_tgt" or property-like "node_amplitude"
        if "_" in target:
            suffix = target.split("_")[-1]

            # Edge targets: accept "src_tgt" where suffix is not "amplitude" or "frequency"
            if suffix not in ["amplitude", "frequency"]:
                try:
                    self.edges[target]  # validate edge exists
                except KeyError as exc:
                    raise ValueError(f"Target '{target}' is not in the setup.") from exc
                self._nodes[name]["target"] = target

            # Property targets: "node_amplitude" or "node_frequency"
            else:
                node_name = target.split("_", 1)[0]
                try:
                    self.nodes[node_name]  # validate node exists
                except KeyError as exc:
                    raise ValueError(f"Target '{node_name}' is not in the setup.") from exc
                self._nodes[name]["target"] = node_name
                self._nodes[name]["target_property"] = suffix
        else:
            # Plain node target
            try:
                self.nodes[target]  # validate node exists
            except KeyError as exc:
                raise ValueError(f"Target '{target}' is not in the setup.") from exc
            self._nodes[name]["target"] = target

    # Validate and store port and direction, when relevant
    if port is not None:
        if port not in ["left", "top", "right", "bottom"]:
            raise ValueError(
                f"Port '{port}' is not recognized. Use 'left', 'top', 'right', or 'bottom'."
            )
        self._nodes[name]["port"] = port

    if direction is not None:
        if direction not in ["in", "out"]:
            raise ValueError(
                f"Direction '{direction}' is not recognized. Use 'in' or 'out'."
            )
        self._nodes[name]["direction"] = direction

    if auxiliary is not None:
        if auxiliary not in [True, False]:
            raise ValueError(
                f"Auxiliary '{auxiliary}' is not recognized. Use True or False."
            )
        self._nodes[name]["auxiliary"] = auxiliary

    # Validate detector references for components that depend on them
    if detector1 is not None:
        if detector1 not in self._nodes:
            raise ValueError(f"Detector1 '{detector1}' is not in the setup.")
        self._nodes[name]["detector1"] = detector1

    if detector2 is not None:
        if detector2 not in self._nodes:
            raise ValueError(f"Detector2 '{detector2}' is not in the setup.")
        self._nodes[name]["detector2"] = detector2


def space(
    self,
    source: str,
    target: str,
    not_optimizable: NotOptimizable = None,
    source_port: str = "right",
    target_port: str = "left",
    **properties: Any,
) -> None:
    if source not in self._nodes:
        raise ValueError(f"Source node '{source}' is not in the setup.")
    if target not in self._nodes:
        raise ValueError(f"Target node '{target}' is not in the setup.")

    if not_optimizable is None:
        not_optimizable = []

    # Merge and validate properties
    properties = {**self.default_properties["space"], **(properties or {})}
    if len(properties) != len(self.default_properties["space"]):
        raise ValueError(
            f"Space has the properties {list(self.default_properties['space'].keys())} "
            f"but received {list(properties.keys())}."
        )

    # Add optimizable parameters for spaces, unless excluded
    edge_name = f"{source}_{target}"
    if not_optimizable is not True:
        self.parameters.extend(
            [
                (edge_name, property_name)
                for property_name in properties.keys()
                if (edge_name, property_name) not in not_optimizable  # type: ignore[operator]
            ]
        )

    # Store edge
    self._edges[(source, target)] = {
        "properties": properties,
        "source_port": source_port,
        "target_port": target_port,
    }
