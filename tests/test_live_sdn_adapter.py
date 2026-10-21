import networkx as nx
import pytest

from live_sdn.routing_adapter import build_live_graph, choose_path, path_outputs


def test_build_live_graph_preserves_directed_ports_and_compacts_no_ids():
    graph = build_live_graph(
        [1, 2, 3],
        [(1, 11, 2, 21), (2, 21, 1, 11), (2, 22, 3, 31), (3, 31, 2, 22)],
    )
    assert graph[1][2]["out_port"] == 11
    assert path_outputs(graph, [1, 2, 3]) == [(1, 11), (2, 22)]


def test_shortest_path_excludes_failed_link_and_has_no_loop():
    links = [
        (1, 11, 2, 21), (2, 21, 1, 11),
        (2, 22, 4, 41), (4, 41, 2, 22),
        (1, 13, 3, 31), (3, 31, 1, 13),
        (3, 34, 4, 43), (4, 43, 3, 34),
    ]
    graph = build_live_graph([1, 2, 3, 4], links, failed_links=[(2, 4)])
    path = choose_path(graph, 1, 4)
    assert path == [1, 3, 4]
    assert len(path) == len(set(path))


def test_congestion_route_prefers_less_utilized_equal_length_path():
    links = [
        (1, 11, 2, 21), (2, 21, 1, 11), (2, 22, 4, 41), (4, 41, 2, 22),
        (1, 13, 3, 31), (3, 31, 1, 13), (3, 34, 4, 43), (4, 43, 3, 34),
    ]
    metrics = {(1, 2): {"utilization": .99}, (2, 1): {"utilization": .99},
               (2, 4): {"utilization": .99}, (4, 2): {"utilization": .99},
               (1, 3): {"utilization": .1}, (3, 1): {"utilization": .1},
               (3, 4): {"utilization": .1}, (4, 3): {"utilization": .1}}
    graph = build_live_graph([1, 2, 3, 4], links, metrics=metrics)
    assert choose_path(graph, 1, 4, congestion_aware=True) == [1, 3, 4]


def test_unknown_utilization_does_not_invent_congestion_cost():
    graph = build_live_graph([1, 2, 3], [(1, 1, 2, 2), (2, 2, 3, 3)])
    assert choose_path(graph, 1, 3, congestion_aware=True) == [1, 2, 3]
    assert all("utilization" not in row for _, _, row in graph.edges(data=True))


def test_no_route_returns_none():
    graph = nx.DiGraph()
    graph.add_nodes_from([1, 2])
    assert choose_path(graph, 1, 2) is None
    assert choose_path(graph, 8, 2) is None

