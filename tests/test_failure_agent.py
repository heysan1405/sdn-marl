import torch

from agents.failure_agent import FailureAgent


def make_state():

    return {
        "node_features": torch.tensor(
            [
                [1.0, 0.20, 0.00, 0.00, 1.0, 0.0, 0.0],
                [2.0, 0.30, 0.20, 0.00, 0.0, 0.0, 0.0],
                [2.0, 0.40, 0.40, 0.00, 0.0, 0.0, 0.0],
                [2.0, 0.60, 0.60, 0.00, 0.0, 0.0, 0.0],
                [2.0, 0.70, 0.80, 0.00, 0.0, 0.0, 0.0],
                [1.0, 0.80, 1.00, 0.00, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),

        "edge_index": torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
            ],
            dtype=torch.long,
        ),

        "edge_features": torch.tensor(
            [
                [0.20, 100.0, 5.0, 0.01, 0.0],
                [0.20, 100.0, 5.0, 0.01, 0.0],
                [0.30, 100.0, 6.0, 0.01, 0.0],
                [0.30, 100.0, 6.0, 0.01, 0.0],
                [0.40, 100.0, 7.0, 0.02, 0.0],
                [0.40, 100.0, 7.0, 0.02, 0.0],
                [0.70, 100.0, 12.0, 0.04, 0.0],
                [0.70, 100.0, 12.0, 0.04, 0.0],
                [0.90, 100.0, 20.0, 0.10, 1.0],
                [0.90, 100.0, 20.0, 0.10, 1.0],
            ],
            dtype=torch.float32,
        ),

        "demand_features": torch.tensor(
            [[40.0]],
            dtype=torch.float32,
        ),
    }


def test_failure_action():

    agent = FailureAgent(
        batch_size=2,
        epsilon_start=0.0,
    )

    action = agent.select_action(
        make_state(),
        training=False,
    )

    assert action in [0, 1, 2]


def test_failure_learning():

    agent = FailureAgent(
        batch_size=2,
        epsilon_start=0.0,
    )

    state = make_state()

    for _ in range(2):

        agent.remember(
            state,
            0,
            1.0,
            state,
            False,
        )

    loss = agent.learn()

    assert loss is not None
    assert loss >= 0.0


def test_failure_mode_restore():

    agent = FailureAgent(
        batch_size=2,
        epsilon_start=0.0,
    )

    agent.online_gnn.train()
    agent.online_network.train()

    assert agent.online_gnn.training
    assert agent.online_network.training

    agent.select_action(
        make_state(),
        training=False,
    )

    assert agent.online_gnn.training
    assert agent.online_network.training