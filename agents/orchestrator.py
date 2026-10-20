"""Transparent safety and conflict coordinator for the specialized agents."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    congestion_action: int
    failure_action: int
    reason: str


class RuleBasedOrchestrator:
    """Prioritize failure recovery and validate agent action proposals.

    It is a deterministic rules baseline, not a learned policy. State uses the
    SDNEnvironment schema: node failed flag at column 6 and edge failed flag
    at column 4.
    """
    CONGESTION_ACTIONS = {0, 1, 2}
    FAILURE_ACTIONS = {0, 1, 2}

    @staticmethod
    def _has_failed_components(state):
        if not isinstance(state, dict):
            raise TypeError("state must be a mapping of environment tensors")
        nodes = state.get("node_features")
        edges = state.get("edge_features")
        if nodes is None or edges is None:
            raise ValueError("state requires node_features and edge_features")
        if getattr(nodes, "ndim", None) != 2 or nodes.shape[1] < 7:
            raise ValueError("node_features must have at least 7 columns")
        if getattr(edges, "ndim", None) != 2 or edges.shape[1] < 5:
            raise ValueError("edge_features must have at least 5 columns")
        return bool((nodes[:, 6] > 0).any().item() or (edges[:, 4] > 0).any().item())

    def resolve(self, state, congestion_proposal, failure_proposal):
        """Return executable actions and an auditable reason."""
        if congestion_proposal not in self.CONGESTION_ACTIONS:
            congestion_proposal = 0
            congestion_invalid = True
        else:
            congestion_invalid = False
        if failure_proposal not in self.FAILURE_ACTIONS:
            failure_proposal = 0
            failure_invalid = True
        else:
            failure_invalid = False

        if self._has_failed_components(state):
            return Decision(0, 1, "failed components present: prioritize reroute recovery")
        reasons = []
        if congestion_invalid or failure_invalid:
            reasons.append("invalid proposal replaced with safe no-op")
        if failure_proposal != 0:
            reasons.append("failure action suppressed: no observed failure")
        if congestion_proposal == 1 and failure_proposal == 1:
            reasons.append("duplicate reroute proposals collapsed")
        return Decision(congestion_proposal, 0, "; ".join(reasons) or "apply congestion proposal")
