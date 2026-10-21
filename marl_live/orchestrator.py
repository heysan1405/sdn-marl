"""Auditable coordinator: dispatch failure recovery or congestion routing."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    action_index: int
    next_node: int
    decision_source: str
    reason: str


class RoutingOrchestrator:
    """Dispatch congestion routing or fuse both learned proposals under failure."""
    FAILURE_POLICY_WEIGHT = 0.5

    def decide(self, observation, congestion_policy=None, failure_policy=None):
        candidates = observation["candidate_nodes"]
        mask = observation["action_mask"]
        features = observation["candidate_features"]
        if not candidates or not any(mask):
            raise RuntimeError("no valid actions remain; drop/hold the packet")
        has_failure = bool(observation.get("has_failure", False))
        if has_failure and congestion_policy is not None and failure_policy is not None:
            try:
                import torch
                with torch.no_grad():
                    congestion_dist, _ = congestion_policy.distribution(observation)
                    failure_dist, _ = failure_policy.distribution(observation)
                    weight = self.FAILURE_POLICY_WEIGHT
                    scores = ((1.0 - weight) * congestion_dist.probs.clamp_min(1e-12).log()
                              + weight * failure_dist.probs.clamp_min(1e-12).log())
                    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=scores.device)
                    action = int(scores.masked_fill(~mask_tensor, float("-inf")).argmax().item())
                if 0 <= action < len(candidates) and bool(mask[action]):
                    return RouteDecision(action, int(candidates[action]), "coordinated_policy_fusion",
                                         "equal log-probability fusion under observed failure")
            except (ValueError, RuntimeError, IndexError, TypeError, AttributeError):
                pass
        policy = failure_policy if has_failure else congestion_policy
        source = "failure_policy" if has_failure else "congestion_policy"
        if policy is not None:
            try:
                action = int(policy.select_action(observation, stochastic=False))
                if 0 <= action < len(candidates) and bool(mask[action]):
                    return RouteDecision(action, int(candidates[action]), source,
                                         "trained policy inference")
            except (ValueError, RuntimeError, IndexError, TypeError):
                pass
        # Explicitly named deterministic safety fallback: maximize destination
        # progress, then prefer lower known utilization and stable node ordering.
        valid = [i for i, allowed in enumerate(mask) if bool(allowed)]
        chosen = max(valid, key=lambda i: (
            float(features[i, 3]),
            float(features[i, 5]),
            -float(features[i, 0]) if float(features[i, 1]) > 0.5 else 0.0,
            -int(candidates[i]),
        ))
        why = "policy unavailable" if policy is None else "invalid policy output"
        return RouteDecision(chosen, int(candidates[chosen]), "safety_fallback", why)
