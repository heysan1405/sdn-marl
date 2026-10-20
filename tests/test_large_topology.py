import networkx as nx
from scalability.large_topology import read_matrix_market, connected_subgraph, route_distance_constrained, distance_decreasing_hop


def test_matrix_market_symmetric_duplicates_and_loops(tmp_path):
    path = tmp_path / "tiny.mtx"
    path.write_text("%MatrixMarket matrix coordinate pattern symmetric\n4 4 4\n2 1\n1 2\n3 2\n4 4\n")
    graph, meta = read_matrix_market(path)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 2
    assert meta["duplicate_edge_entries"] == 1
    assert meta["self_loops"] == 1


def test_seeded_subgraph_is_connected_and_preserves_mapping():
    graph = nx.path_graph([10, 20, 30, 40, 50])
    first, mapping = connected_subgraph(graph, 4, 7)
    second, mapping2 = connected_subgraph(graph, 4, 7)
    assert nx.is_connected(first)
    assert set(first.edges) == set(second.edges)
    assert mapping == mapping2
    assert set(mapping["local_to_original"]) <= set(graph)


def test_forwarding_strictly_decreases_destination_distance():
    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (0, 3)])
    distances = nx.single_source_shortest_path_length(graph, 3)
    for node in graph:
        nxt = distance_decreasing_hop(graph, distances, node, 3)
        if nxt is not None:
            assert distances[nxt] < distances[node]
    assert route_distance_constrained(graph, 0, 3) == {"status": "delivered", "hops": 1}

