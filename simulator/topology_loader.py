
"""
Topology Zoo GML Loader
=======================

Loads arbitrary Internet Topology Zoo GML files into NetworkX graphs.

Features:
- Supports different topology sizes
- Handles duplicate edges found in some Zoo GML files
- Extracts link capacity from LinkSpeedRaw / LinkSpeed
- Converts Gbps/Mbps/Kbps correctly
- Maps common legacy link types such as OC-192
- Preserves node metadata
- Adds explicit simulation defaults for missing weight and delay
- Provides a command-line interface

Usage:
    python -m simulator.topology_loader --topology Abilene
    python -m simulator.topology_loader --topology GEANT
    python -m simulator.topology_loader --topology NSFNET
    python -m simulator.topology_loader --file path/to/topology.gml
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GML_DIRECTORY = PROJECT_ROOT / "InternetTopologyZoo" / "gml"


# ---------------------------------------------------------------------------
# Topology name resolution
# ---------------------------------------------------------------------------

TOPOLOGY_ALIASES = {
    "abilene": "Abilene.gml",
    "geant": "Geant.gml",
    "nsfnet": "Nsfnet.gml",
}


# ---------------------------------------------------------------------------
# Default simulation parameters
# ---------------------------------------------------------------------------

DEFAULT_CAPACITY_MBPS = 1000.0
DEFAULT_WEIGHT = 1.0
DEFAULT_DELAY_MS = 1.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float if possible."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_units(units: Any) -> str:
    """Normalize bandwidth unit strings."""
    if units is None:
        return ""

    return str(units).strip().lower().replace("bps", "b")


def _convert_bandwidth_to_mbps(
    value: float,
    units: str,
) -> Optional[float]:
    """
    Convert bandwidth to Mbps.

    Supported examples:
        G / Gbps -> Gbps
        M / Mbps -> Mbps
        K / Kbps -> Kbps
        bps      -> bps
    """

    units = _normalize_units(units)

    if units in {"g", "gb", "gbps"}:
        return value * 1000.0

    if units in {"m", "mb", "mbps"}:
        return value

    if units in {"k", "kb", "kbps"}:
        return value / 1000.0

    if units in {"b", "bps"}:
        return value / 1_000_000.0

    return None


def _extract_capacity_mbps(
    edge_data: Dict[str, Any],
) -> Tuple[float, str]:
    """
    Extract link capacity in Mbps.

    Priority:

    1. LinkSpeedRaw
       Usually represented in bits per second.

    2. LinkSpeed + LinkSpeedUnits

    3. Parse LinkLabel if it contains a bandwidth.

    4. Recognize common legacy link types.

    5. Fall back to DEFAULT_CAPACITY_MBPS.

    Returns:
        (capacity_mbps, source_description)
    """

    # ------------------------------------------------------------------
    # 1. LinkSpeedRaw
    # ------------------------------------------------------------------

    raw = _safe_float(edge_data.get("LinkSpeedRaw"))

    if raw is not None and raw > 0:

        # Topology Zoo stores LinkSpeedRaw as bits per second.
        capacity_mbps = raw / 1_000_000.0

        return capacity_mbps, "LinkSpeedRaw"


    # ------------------------------------------------------------------
    # 2. LinkSpeed + LinkSpeedUnits
    # ------------------------------------------------------------------

    speed = _safe_float(edge_data.get("LinkSpeed"))
    units = edge_data.get("LinkSpeedUnits")

    if speed is not None and speed > 0:

        converted = _convert_bandwidth_to_mbps(speed, units)

        if converted is not None and converted > 0:
            return converted, "LinkSpeed + LinkSpeedUnits"


    # ------------------------------------------------------------------
    # 3. Try LinkLabel
    # ------------------------------------------------------------------

    label = edge_data.get("LinkLabel")

    if label is not None:

        text = str(label).strip().lower()

        # Examples:
        #   "2.5 Gbps"
        #   "10 Gbps"
        #   "100 Mbps"
        #   "155 Mbps"

        match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*(gbps|gbit/s|g|mbps|mbit/s|m|kbps|kbit/s|k)",
            text,
        )

        if match:

            value = float(match.group(1))
            unit = match.group(2)

            converted = _convert_bandwidth_to_mbps(
                value,
                unit,
            )

            if converted is not None and converted > 0:
                return converted, "LinkLabel"


    # ------------------------------------------------------------------
    # 4. Legacy optical link types
    # ------------------------------------------------------------------

    link_type = str(
        edge_data.get("LinkType", "")
    ).strip().upper()

    optical_capacities = {

        # OC-3 = 155.52 Mbps
        "OC-3": 155.52,

        # OC-12 = 622.08 Mbps
        "OC-12": 622.08,

        # OC-48 = 2.48832 Gbps
        "OC-48": 2488.32,

        # OC-192 = 9.95328 Gbps
        "OC-192": 9953.28,

        # OC-768 = 39.81312 Gbps
        "OC-768": 39813.12,
    }

    if link_type in optical_capacities:

        return (
            optical_capacities[link_type],
            f"LinkType {link_type}",
        )


    # ------------------------------------------------------------------
    # 5. Generic fallback
    # ------------------------------------------------------------------

    return (
        DEFAULT_CAPACITY_MBPS,
        "default",
    )


# ---------------------------------------------------------------------------
# GML duplicate-edge detection
# ---------------------------------------------------------------------------

def _has_duplicate_edges(path: Path) -> bool:
    """
    Detect duplicate source-target pairs in a GML file.

    NetworkX Graph does not allow duplicate edges during GML parsing,
    so some Topology Zoo files need custom handling.
    """

    try:

        source_target_pairs = set()

        current_source = None
        current_target = None

        inside_edge = False

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            for line in file:

                stripped = line.strip()

                if stripped.startswith("edge ["):
                    inside_edge = True
                    current_source = None
                    current_target = None
                    continue

                if inside_edge:

                    if stripped.startswith("source "):

                        current_source = stripped.split(
                            None,
                            1,
                        )[1]

                    elif stripped.startswith("target "):

                        current_target = stripped.split(
                            None,
                            1,
                        )[1]

                    elif stripped == "]":

                        if (
                            current_source is not None
                            and current_target is not None
                        ):

                            pair = tuple(
                                sorted(
                                    (
                                        current_source,
                                        current_target,
                                    )
                                )
                            )

                            if pair in source_target_pairs:
                                return True

                            source_target_pairs.add(pair)

                        inside_edge = False

        return False

    except OSError:

        return False


# ---------------------------------------------------------------------------
# Manual duplicate-tolerant GML parser
# ---------------------------------------------------------------------------

def _read_gml_duplicate_tolerant(
    path: Path,
) -> nx.Graph:
    """
    Read a GML file while tolerating duplicate edges.

    Duplicate edges are merged into a single undirected edge.

    The first occurrence supplies the edge attributes.
    """

    graph = nx.Graph()

    nodes = []
    edges = []

    current_node = None
    current_edge = None

    section = None

    # Simple stack-free parser for the standard Zoo GML format.
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        lines = file.readlines()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        # --------------------------------------------------------------
        # NODE
        # --------------------------------------------------------------

        if line == "node [":

            current_node = {}
            section = "node"

            i += 1
            continue


        # --------------------------------------------------------------
        # EDGE
        # --------------------------------------------------------------

        if line == "edge [":

            current_edge = {}
            section = "edge"

            i += 1
            continue


        # --------------------------------------------------------------
        # Closing bracket
        # --------------------------------------------------------------

        if line == "]":

            if section == "node" and current_node is not None:

                if "id" in current_node:

                    nodes.append(current_node)

                current_node = None


            elif section == "edge" and current_edge is not None:

                if (
                    "source" in current_edge
                    and "target" in current_edge
                ):

                    edges.append(current_edge)

                current_edge = None

            section = None

            i += 1
            continue


        # --------------------------------------------------------------
        # Parse key/value
        # --------------------------------------------------------------

        if section in {"node", "edge"} and line:

            parts = line.split(None, 1)

            if len(parts) == 2:

                key = parts[0]
                value = parts[1].strip()

                # Remove quoted strings.
                if (
                    len(value) >= 2
                    and value.startswith('"')
                    and value.endswith('"')
                ):
                    value = value[1:-1]

                # Numeric conversion.
                else:

                    try:

                        if "." in value or "e" in value.lower():

                            value = float(value)

                        else:

                            value = int(value)

                    except ValueError:

                        pass

                if section == "node":
                    current_node[key] = value

                else:
                    current_edge[key] = value

        i += 1


    # ------------------------------------------------------------------
    # Add nodes
    # ------------------------------------------------------------------

    for node_data in nodes:

        node_id = node_data.get("id")

        graph.add_node(
            node_id,
            **node_data,
        )


    # ------------------------------------------------------------------
    # Add edges
    # ------------------------------------------------------------------

    for edge_data in edges:

        source = edge_data.get("source")
        target = edge_data.get("target")

        if source is None or target is None:
            continue

        # Ignore exact duplicate source-target pairs.
        if graph.has_edge(source, target):

            existing = graph[source][target]

            # Preserve useful attributes that the first edge may not have.
            for key, value in edge_data.items():

                if key not in existing:
                    existing[key] = value

            continue

        graph.add_edge(
            source,
            target,
            **edge_data,
        )


    return graph


# ---------------------------------------------------------------------------
# Topology loader
# ---------------------------------------------------------------------------

class TopologyLoader:
    """Loads and normalizes Topology Zoo GML files."""

    def __init__(
        self,
        topology_directory: Optional[Path] = None,
    ):

        self.topology_directory = (
            Path(topology_directory)
            if topology_directory is not None
            else GML_DIRECTORY
        )


    def resolve_topology_file(
        self,
        topology_name: str,
    ) -> Path:

        requested = topology_name.strip()

        # Direct path.
        possible_path = Path(requested)

        if possible_path.exists():
            return possible_path.resolve()


        # Alias.
        alias_key = requested.lower()

        if alias_key in TOPOLOGY_ALIASES:

            path = (
                self.topology_directory
                / TOPOLOGY_ALIASES[alias_key]
            )

            if path.exists():
                return path.resolve()


        # Case-insensitive filename search.
        if self.topology_directory.exists():

            requested_filename = requested.lower()

            if not requested_filename.endswith(".gml"):
                requested_filename += ".gml"

            for file in self.topology_directory.glob("*.gml"):

                if file.name.lower() == requested_filename:

                    return file.resolve()


        raise FileNotFoundError(
            f"Topology '{topology_name}' was not found in "
            f"{self.topology_directory}"
        )


    def load(
        self,
        topology_name: Optional[str] = None,
        topology_file: Optional[str] = None,
    ) -> nx.Graph:

        # --------------------------------------------------------------
        # Resolve path
        # --------------------------------------------------------------

        if topology_file:

            path = Path(topology_file).expanduser().resolve()

        elif topology_name:

            path = self.resolve_topology_file(
                topology_name
            )

        else:

            raise ValueError(
                "Provide either topology_name or topology_file."
            )


        if not path.exists():

            raise FileNotFoundError(
                f"GML file does not exist:\n{path}"
            )


        print("=" * 70)
        print("LOADING TOPOLOGY")
        print("=" * 70)
        print(f"File: {path}")


        # --------------------------------------------------------------
        # Detect duplicate edges
        # --------------------------------------------------------------

        has_duplicates = _has_duplicate_edges(path)


        # --------------------------------------------------------------
        # Read graph
        # --------------------------------------------------------------

        if has_duplicates:

            print(
                "[INFO] Duplicate edges detected."
            )

            print(
                "[INFO] Using duplicate-edge tolerant GML parser."
            )

            graph = _read_gml_duplicate_tolerant(path)

        else:

            try:

                graph = nx.read_gml(
                    path,
                    label=None,
                )

            except Exception:

                # Some GML files can still contain formatting quirks.
                print(
                    "[INFO] Standard GML loading failed."
                )

                print(
                    "[INFO] Using duplicate-edge tolerant parser."
                )

                graph = _read_gml_duplicate_tolerant(path)


        # --------------------------------------------------------------
        # Normalize node IDs
        # --------------------------------------------------------------

        graph = self._normalize_node_ids(graph)


        # --------------------------------------------------------------
        # Normalize edge attributes
        # --------------------------------------------------------------

        self._normalize_edges(graph)


        # --------------------------------------------------------------
        # Validate
        # --------------------------------------------------------------

        if len(graph) == 0:

            raise RuntimeError(
                f"Topology contains no nodes:\n{path}"
            )


        if not nx.is_connected(graph):

            print(
                "[WARNING] Topology is disconnected."
            )


        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        print(
            f"Nodes: {graph.number_of_nodes()}"
        )

        print(
            f"Edges: {graph.number_of_edges()}"
        )

        print(
            f"Connected: {nx.is_connected(graph)}"
        )

        print("=" * 70)


        return graph


    # ------------------------------------------------------------------
    # Node normalization
    # ------------------------------------------------------------------

    def _normalize_node_ids(
        self,
        graph: nx.Graph,
    ) -> nx.Graph:

        mapping = {}

        nodes = list(graph.nodes())

        for index, node in enumerate(nodes):

            mapping[node] = index


        graph = nx.relabel_nodes(
            graph,
            mapping,
        )


        for index, node in enumerate(graph.nodes()):

            data = graph.nodes[node]

            data["original_id"] = str(
                nodes[index]
            )

            data["id"] = index


        return graph


    # ------------------------------------------------------------------
    # Edge normalization
    # ------------------------------------------------------------------

    def _normalize_edges(
        self,
        graph: nx.Graph,
    ):

        for source, target, data in graph.edges(
            data=True
        ):

            # ----------------------------------------------------------
            # Capacity
            # ----------------------------------------------------------

            capacity, capacity_source = (
                _extract_capacity_mbps(data)
            )

            data["capacity_mbps"] = float(
                capacity
            )

            data["capacity_source"] = (
                capacity_source
            )


            # ----------------------------------------------------------
            # Weight
            # ----------------------------------------------------------

            existing_weight = _safe_float(
                data.get("weight")
            )

            if (
                existing_weight is not None
                and existing_weight > 0
            ):

                data["weight"] = existing_weight
                data["weight_source"] = "GML"

            else:

                data["weight"] = DEFAULT_WEIGHT
                data["weight_source"] = "default"


            # ----------------------------------------------------------
            # Delay
            # ----------------------------------------------------------

            existing_delay = _safe_float(
                data.get("delay")
            )

            if (
                existing_delay is not None
                and existing_delay >= 0
            ):

                data["delay_ms"] = existing_delay
                data["delay_source"] = "GML"

            else:

                data["delay_ms"] = DEFAULT_DELAY_MS
                data["delay_source"] = "default"


            # ----------------------------------------------------------
            # Dynamic simulator values
            # ----------------------------------------------------------

            data["utilization"] = 0.0
            data["traffic_mbps"] = 0.0
            data["packet_loss"] = 0.0


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def load_topology(
    topology_name: Optional[str] = None,
    topology_file: Optional[str] = None,
) -> nx.Graph:

    loader = TopologyLoader()

    return loader.load(
        topology_name=topology_name,
        topology_file=topology_file,
    )


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def print_topology_details(
    graph: nx.Graph,
):

    print()
    print("=" * 70)
    print("NODES")
    print("=" * 70)

    for node, data in graph.nodes(
        data=True
    ):

        print(
            f"{node}: {data}"
        )


    print()
    print("=" * 70)
    print("EDGES")
    print("=" * 70)

    for source, target, data in graph.edges(
        data=True
    ):

        print(
            f"{source} <-> {target} | "
            f"capacity={data['capacity_mbps']:.2f} Mbps | "
            f"weight={data['weight']:.2f} | "
            f"delay={data['delay_ms']:.2f} ms | "
            f"capacity_source={data['capacity_source']}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Load an Internet Topology Zoo GML topology."
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--topology",
        type=str,
        help="Topology name, e.g. Abilene, GEANT, NSFNET",
    )

    group.add_argument(
        "--file",
        type=str,
        help="Path to a GML topology file",
    )

    args = parser.parse_args()


    graph = load_topology(
        topology_name=args.topology,
        topology_file=args.file,
    )


    print_topology_details(graph)


if __name__ == "__main__":
    main()
