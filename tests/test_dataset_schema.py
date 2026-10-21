import networkx as nx
import pytest

from training.dataset_schema import candidate_features, link_usage_by_edge
from training.dataset_parser import DatasetParser


def test_link_usage_uses_row_major_node_pair_indices_not_edge_iteration_order():
    graph = nx.DiGraph()
    graph.add_nodes_from(range(3))
    # Deliberately insert edges in an order different from a row-major matrix.
    graph.add_edge(0, 2, bandwidth=25000, weight=1)
    graph.add_edge(0, 1, bandwidth=10000, weight=2)
    graph.add_edge(2, 0, bandwidth=40000, weight=1)
    matrix = [None] * 9
    matrix[0 * 3 + 2] = [0.2, 0.01, 800, 0.3, 0.02, 2, 4, 900]
    matrix[0 * 3 + 1] = [0.4, 0.03, 1000, 0.5, 0.04, 6, 8, 1100]
    matrix[2 * 3 + 0] = [0.6, 0.05, 1200, 0.7, 0.06, 10, 12, 1300]

    state = link_usage_by_edge(graph, matrix)
    assert state[(0, 2)]["metrics"][0] == 0.2
    assert state[(0, 1)]["metrics"][0] == 0.4
    assert state[(2, 0)]["metrics"][0] == 0.6
    assert candidate_features(state[(0, 2)], 3) == [25000, 1, 0.2, 0.01, 800, 0.3, 0.02, 2, 4, 900, 3]


def test_link_usage_rejects_missing_arc_measurements():
    graph = nx.DiGraph([(0, 1)])
    graph.add_nodes_from([0, 1])
    with pytest.raises(ValueError, match="no linkUsage measurement"):
        link_usage_by_edge(graph, [None] * 4)


def test_stability_parser_keeps_status_and_simulator_metadata():
    row = DatasetParser.parse_stability_field(
        "38725;0.000911112;OK;Mem(KB):320164;Time(s):244"
    )
    assert row["duration"] == 38725
    assert row["intensity"] == pytest.approx(0.000911112)
    assert row["status"] == "OK"
    assert row["metadata"] == ["Mem(KB):320164", "Time(s):244"]
