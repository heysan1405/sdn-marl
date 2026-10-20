import networkx as nx

from simulator.environment import SDNEnvironment


def make_env(tmp_path, traffic_scale=1.0):
    graph = nx.Graph([(0, 1), (1, 3), (0, 2), (2, 3), (1, 2)])
    path = tmp_path / "diamond.gml"
    nx.write_gml(graph, path)
    env = SDNEnvironment(str(path), num_demands=2, max_steps=2,
                         random_seed=13, traffic_scale=traffic_scale)
    env.reset(seed=13)
    return env


def test_traffic_scale_is_applied_to_generated_offered_load(tmp_path):
    base = make_env(tmp_path, 1.0)
    stressed = make_env(tmp_path, 8.0)
    assert [d["traffic_mbps"] * 8 for d in base.demands] == [
        d["traffic_mbps"] for d in stressed.demands
    ]


def test_undirected_links_are_serialized_in_both_message_directions(tmp_path):
    env = make_env(tmp_path)
    state = env._build_state()
    edge_index = state["edge_index"]
    edge_features = state["edge_features"]

    assert edge_index.shape[1] == 2 * env.graph.number_of_edges()
    directed = {tuple(edge): i for i, edge in enumerate(edge_index.T.tolist())}
    for (source, target), index in directed.items():
        reverse_index = directed[(target, source)]
        assert (edge_features[index] == edge_features[reverse_index]).all()


def test_state_dimensions_and_action_masks_work_on_variable_graph_sizes(tmp_path):
    from agents.congestion_agent import CongestionAgent

    for name, edges, expected_nodes in (
        ("small", [(0, 1), (1, 2)], 3),
        ("larger", [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4),
    ):
        path = tmp_path / f"{name}.gml"
        nx.write_gml(nx.Graph(edges), path)
        env = SDNEnvironment(str(path), num_demands=1, max_steps=1, random_seed=21)
        state = env.reset(seed=21)
        assert state["node_features"].shape == (expected_nodes, 7)
        assert state["edge_features"].shape == (2 * len(edges), 5)
        agent = CongestionAgent(device="cpu")
        action = agent.select_action(state, training=False)
        assert action in (env.CONGESTION_MAINTAIN, env.CONGESTION_REROUTE, env.CONGESTION_RATE_LIMIT)


def test_rate_limit_changes_offered_load_but_keeps_original_denominator(tmp_path):
    env = make_env(tmp_path, 5.0)
    before = env.simulator.demands[0]["traffic_mbps"]
    original = env.simulator.original_traffic[0]
    result = env._apply_congestion_action(env.CONGESTION_RATE_LIMIT, env.demands[0])
    assert result["applied"]
    assert env.simulator.demands[0]["traffic_mbps"] == before * env.rate_limit_factor
    assert env.simulator.original_traffic[0] == original


def test_failure_reroute_repairs_all_broken_installed_demands(tmp_path):
    env = make_env(tmp_path)
    # Install two flows over the same first-choice route.
    demands = [
        {"src": 0, "dst": 3, "traffic_mbps": 100.0},
        {"src": 0, "dst": 3, "traffic_mbps": 120.0},
    ]
    env.demands = demands
    env.simulator.install_demands(demands)
    env.current_demand_index = 0
    original_path = env.simulator.active_paths[0]
    broken = tuple(original_path[0:2])
    env.inject_link_failure(*broken)
    assert any(not env.simulator._path_is_usable(path)
               for path in env.simulator.active_paths.values())

    outcome = env._apply_failure_action(env.FAILURE_REROUTE, demands[0])
    assert outcome["applied"]
    assert outcome["rerouted_demands"] >= 1
    assert all(env.simulator._path_is_usable(path)
               for path in env.simulator.active_paths.values())


def test_failure_isolate_action_preserves_and_repairs_affected_demands(tmp_path):
    env = make_env(tmp_path)
    demands = [
        {"src": 0, "dst": 3, "traffic_mbps": 100.0},
        {"src": 0, "dst": 3, "traffic_mbps": 120.0},
    ]
    env.demands = demands
    env.simulator.install_demands(demands)
    env.current_demand_index = 0
    path = env.simulator.active_paths[0]
    env.inject_link_failure(*path[:2])

    outcome = env._apply_failure_action(env.FAILURE_ISOLATE, demands[0])
    assert outcome["action"] == "isolate_and_reroute"
    assert len(env.simulator.active_paths) == 2
    assert all(env.simulator._path_is_usable(active)
               for active in env.simulator.active_paths.values())
    assert outcome["result"]["affected_demands"]
