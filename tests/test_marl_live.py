import numpy as np
import networkx as nx
import pytest

from marl_live.environment import RoutingEnv, make_training_graph
from marl_live.features import candidate_observation
from marl_live.orchestrator import RoutingOrchestrator
from marl_live.policy import RoutingPolicy
from marl_live.routing import build_policy_route


def test_environment_reset_is_seeded_and_observation_schema_is_variable_degree():
    env = RoutingEnv()
    first = env.reset(123, "congestion")
    second = env.reset(123, "congestion")
    assert first["candidate_features"].shape[1] == 6
    np.testing.assert_array_equal(first["candidate_features"], second["candidate_features"])
    assert first["candidate_nodes"] == second["candidate_nodes"]
    assert first["candidate_features"].shape[0] == len(first["action_mask"])


def test_shared_feature_builder_uses_unknown_for_missing_live_utilization():
    graph = make_training_graph(4)
    for _, _, edge in graph.edges(data=True):
        edge.pop("background_mbps", None)
        edge.pop("capacity_mbps", None)
        edge["utilization"] = None
    obs = candidate_observation(graph, 0, 7, {0})
    assert np.all(obs["candidate_features"][:, 0] == 0.0)
    assert np.all(obs["candidate_features"][:, 1] == 0.0)
    assert np.all(obs["candidate_features"][:, 2] == 0.0)


def test_failed_link_cannot_be_selected_and_route_is_loop_free():
    env = RoutingEnv()
    obs = env.reset(22, "failure", agent="failure")
    failed = np.flatnonzero(obs["candidate_features"][:, 4] > 0.5)
    assert len(failed) == 1
    assert not obs["action_mask"][failed[0]]
    with pytest.raises(ValueError, match="failed"):
        env.step(int(failed[0]))
    done = False
    while not done:
        obs = env.observe()
        healthy = nx.Graph((u, v) for u, v, d in env.graph.edges(data=True) if not d["is_failed"])
        route = nx.shortest_path(healthy, env.current, env.destination)
        action = obs["candidate_nodes"].index(route[1])
        obs, _, done, info = env.step(action)
    assert info["delivered"]
    assert len(info["path"]) == len(set(info["path"]))


def test_congestion_reward_penalizes_selected_loaded_link_without_reducing_demand():
    env = RoutingEnv()
    obs = env.reset(101, "congestion", agent="congestion")
    valid = np.flatnonzero(obs["action_mask"])
    utilities = obs["candidate_features"][valid, 0]
    hot = int(valid[int(np.argmax(utilities))])
    cool = int(valid[int(np.argmin(utilities))])
    hot_obs = env.reset(101, "congestion", agent="congestion")
    _, hot_reward, _, _ = env.step(hot)
    env.reset(101, "congestion", agent="congestion")
    _, cool_reward, _, _ = env.step(cool)
    assert env.config.demand_mbps == 25.0
    if utilities.max() > utilities.min():
        assert hot_reward <= cool_reward


def test_orchestrator_dispatches_by_failure_context_and_labels_fallback():
    class Fixed:
        def __init__(self, action):
            self.action = action
        def select_action(self, observation, stochastic=False):
            return self.action

    env = RoutingEnv()
    obs = env.reset(8, "normal")
    decision = RoutingOrchestrator().decide(obs, Fixed(0), Fixed(1))
    assert decision.decision_source == "congestion_policy"
    obs = env.reset(8, "failure")
    decision = RoutingOrchestrator().decide(obs, Fixed(0), Fixed(1))
    assert decision.decision_source == "failure_policy"
    decision = RoutingOrchestrator().decide(obs, None, None)
    assert decision.decision_source == "safety_fallback"


def test_checkpoint_metadata_and_inference_round_trip(tmp_path):
    policy = RoutingPolicy(hidden_dim=16)
    path = tmp_path / "policy.pt"
    policy.save(path, {"seed": 2})
    loaded = RoutingPolicy.load(path)
    obs = RoutingEnv().reset(1, "normal")
    action = loaded.select_action(obs)
    assert obs["action_mask"][action]


def test_policy_to_route_blocks_loops_and_reports_safety_fallback():
    class FirstCandidate:
        def select_action(self, observation, stochastic=False):
            return 0

    env = RoutingEnv()
    env.reset(19, "normal")
    result = build_policy_route(env.graph, 0, 7, RoutingOrchestrator(),
                                FirstCandidate(), FirstCandidate(), max_hops=12)
    assert result is not None
    path, sources = result
    assert path[0] == 0 and path[-1] == 7
    assert len(path) == len(set(path))
    assert "safety_fallback" in sources
