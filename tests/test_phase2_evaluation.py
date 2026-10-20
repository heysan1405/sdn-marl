from training.evaluate_dqn import used_links_to_fail


class _Simulator:
    active_paths = {
        0: ["a", "b", "c"],
        1: ["a", "b", "d"],
        2: ["a", "b", "e"],
        3: ["c", "d"],
    }


class _Environment:
    simulator = _Simulator()


def test_multiple_failure_selector_returns_distinct_highest_load_links():
    selected = used_links_to_fail(_Environment(), count=2)
    assert selected == [("a", "b"), ("b", "c")]
    assert len(set(selected)) == 2


def test_failure_selector_is_deterministic_for_same_state():
    first = used_links_to_fail(_Environment(), count=3)
    second = used_links_to_fail(_Environment(), count=3)
    assert first == second
