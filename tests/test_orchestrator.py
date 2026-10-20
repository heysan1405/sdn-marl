import torch

from agents.orchestrator import RuleBasedOrchestrator


def state(failed_node=False, failed_edge=False):
    nodes = torch.zeros((2, 7))
    edges = torch.zeros((1, 5))
    nodes[0, 6] = float(failed_node)
    edges[0, 4] = float(failed_edge)
    return {"node_features": nodes, "edge_features": edges}


def test_normal_state_passes_congestion_and_suppresses_failure_action():
    decision = RuleBasedOrchestrator().resolve(state(), 2, 2)
    assert decision.congestion_action == 2
    assert decision.failure_action == 0
    assert "no observed failure" in decision.reason


def test_failure_state_prioritizes_reroute_over_congestion():
    for observation in (state(failed_node=True), state(failed_edge=True)):
        decision = RuleBasedOrchestrator().resolve(observation, 2, 0)
        assert (decision.congestion_action, decision.failure_action) == (0, 1)
        assert "prioritize reroute" in decision.reason


def test_invalid_proposals_fall_back_to_safe_noop():
    decision = RuleBasedOrchestrator().resolve(state(), 99, -1)
    assert (decision.congestion_action, decision.failure_action) == (0, 0)
    assert "invalid proposal" in decision.reason
