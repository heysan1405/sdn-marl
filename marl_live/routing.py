"""Pure policy-to-next-hop route construction, independent of Ryu/OpenFlow."""
from marl_live.features import candidate_observation


def build_policy_route(graph, source, destination, orchestrator, congestion_policy=None,
                       failure_policy=None, max_hops=32, on_decision=None):
    """Repeatedly ask the coordinator for a safe next hop and return its provenance."""
    if source not in graph or destination not in graph:
        return None
    path, visited, sources = [source], {source}, []
    has_failure = any(bool(data.get("is_failed", False))
                      for _, _, data in graph.edges(data=True))
    while path[-1] != destination and len(path) - 1 < max_hops:
        observation = candidate_observation(graph, path[-1], destination, visited,
                                            has_failure=has_failure)
        try:
            decision = orchestrator.decide(observation, congestion_policy, failure_policy)
        except RuntimeError:
            return None
        if on_decision is not None:
            on_decision(decision, observation)
        next_node = decision.next_node
        if (next_node in visited or not graph.has_edge(path[-1], next_node)
                or bool(graph[path[-1]][next_node].get("is_failed", False))):
            return None
        path.append(next_node)
        visited.add(next_node)
        sources.append(decision.decision_source)
    return (path, sources) if path[-1] == destination else None
