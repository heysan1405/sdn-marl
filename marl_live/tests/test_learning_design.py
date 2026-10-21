import networkx as nx
import numpy as np
import torch
from torch.distributions import Categorical

from marl_live.config import RoutingConfig
from marl_live.environment import RoutingEnv, make_training_graph
from marl_live.features import candidate_observation
from marl_live.orchestrator import RoutingOrchestrator


def test_mask_removes_locally_valid_moves_that_dead_end_behind_visited_nodes():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (2, 1), (1, 4), (2, 3)])
    for u, v in graph.edges:
        graph[u][v].update(is_failed=False, capacity_mbps=100.0,
                           background_mbps=0.0, utilization=0.0)
    obs = candidate_observation(graph, 0, 4, visited={0, 1})
    assert obs["candidate_nodes"] == [1, 2]
    assert obs["action_mask"].tolist() == [False, False]


def test_failed_and_visited_masks_match_environment_and_policy_distribution():
    env = RoutingEnv()
    env.reset(832, "normal", agent="failure")
    failed_neighbor = env.graph.neighbors(env.current).__iter__().__next__()
    env.graph[env.current][failed_neighbor]["is_failed"] = True
    env.failed_edges.add(env._edge_key(env.current, failed_neighbor))
    obs = env.observe()
    failed = np.flatnonzero(obs["candidate_features"][:, 4] > 0.5)
    assert len(failed) == 1
    assert not obs["action_mask"][failed[0]]
    failed_mask = obs["candidate_features"][:, 4] > 0.5
    assert not np.any(failed_mask & obs["action_mask"])


def test_congestion_reward_is_the_negative_weighted_edge_cost_and_failure_reward_is_util_independent():
    env = RoutingEnv()
    obs = env.reset(204, "normal", agent="congestion")
    i = int(np.flatnonzero(obs["action_mask"])[0])
    u, v = env.current, obs["candidate_nodes"][i]
    util = env.graph[u][v]["background_mbps"] / env.graph[u][v]["capacity_mbps"]
    _, reward, _, _ = env.step(i)
    assert np.isclose(reward, -(env.config.hop_cost + env.config.congestion_weight * util))

    fail_env = RoutingEnv()
    obs = fail_env.reset(205, "failure", agent="failure")
    i = int(np.flatnonzero(obs["action_mask"])[0])
    u, v = fail_env.current, obs["candidate_nodes"][i]
    fail_env.graph[u][v]["background_mbps"] += 1000.0
    _, failure_reward, _, _ = fail_env.step(i)
    clean_env = RoutingEnv()
    clean_obs = clean_env.reset(205, "failure", agent="failure")
    clean_i = int(np.flatnonzero(clean_obs["action_mask"])[0])
    _, clean_reward, _, _ = clean_env.step(clean_i)
    assert failure_reward == clean_reward


def test_randomized_graphs_are_seeded_and_training_density_varies():
    a = make_training_graph(5, .35)
    b = make_training_graph(5, .35)
    c = make_training_graph(5, 1.0)
    assert set(a.edges) == set(b.edges)
    assert all(a[u][v]["capacity_mbps"] == b[u][v]["capacity_mbps"] for u, v in a.edges)
    assert set(a.edges) != set(c.edges)


def test_weighted_cost_progress_uses_only_nonvisited_reachable_routes():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 4), (0, 2), (2, 3), (3, 4), (3, 1)])
    for u, v in graph.edges:
        graph[u][v].update(is_failed=False, capacity_mbps=100.0,
                           background_mbps=0.0, utilization=0.0)
    # The cheap-looking detour via 1 cannot be used because 1 was visited.
    graph[0][1]["is_failed"] = True
    graph[0][2]["utilization"] = .1
    graph[2][3]["utilization"] = .1
    graph[3][4]["utilization"] = .1
    obs = candidate_observation(graph, 0, 4, visited={0, 1})
    idx2 = obs["candidate_nodes"].index(2)
    assert obs["action_mask"][idx2]
    assert np.isfinite(obs["candidate_features"][idx2, 5])


def test_orchestrator_fuses_both_agents_during_failures():
    class MockPolicy:
        def __init__(self, probs):
            self.probs = probs
        def distribution(self, observation):
            p = torch.tensor(self.probs, dtype=torch.float32)
            p = p * torch.as_tensor(observation["action_mask"], dtype=torch.float32)
            p = p / p.sum()
            return Categorical(probs=p), torch.tensor(0.0)
        def select_action(self, observation, stochastic=False):
            return int(torch.as_tensor(self.probs).argmax())

    obs = RoutingEnv().reset(832, "failure")
    valid_count = int(obs["action_mask"].sum())
    # Build probabilities that favor different valid choices and zero invalid ones.
    valid = np.flatnonzero(obs["action_mask"])
    assert valid_count >= 2
    congestion, failure = np.zeros(len(obs["action_mask"])), np.zeros(len(obs["action_mask"]))
    congestion[valid[0]], congestion[valid[1]] = .6, .4
    failure[valid[0]], failure[valid[1]] = .1, .9
    decision = RoutingOrchestrator().decide(
        obs, MockPolicy(congestion.tolist()), MockPolicy(failure.tolist()))
    assert decision.decision_source == "coordinated_policy_fusion"
    assert decision.action_index in valid
