"""RL agents for SDN-MARL, loaded lazily to keep utilities lightweight."""

__all__ = ["ReplayBuffer", "CongestionAgent", "FailureAgent", "RuleBasedOrchestrator"]


def __getattr__(name):
    if name == "ReplayBuffer":
        from .replay_buffer import ReplayBuffer
        return ReplayBuffer
    if name == "CongestionAgent":
        from .congestion_agent import CongestionAgent
        return CongestionAgent
    if name == "FailureAgent":
        from .failure_agent import FailureAgent
        return FailureAgent
    if name == "RuleBasedOrchestrator":
        from .orchestrator import RuleBasedOrchestrator
        return RuleBasedOrchestrator
    raise AttributeError(name)
