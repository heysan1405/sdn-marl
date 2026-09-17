import torch

from agents.congestion_agent import CongestionAgent


def make_state():

    return {
        "node_features": torch.tensor(
            [
                [1.0, 0.20, 0.00, 0.00, 1.0, 0.0, 0.0],
                [2.0, 0.30, 0.25, 0.00, 0.0, 0.0, 0.0],
                [2.0, 0.50, 0.50, 0.00, 0.0, 0.0, 0.0],
                [2.0, 0.70, 0.75, 0.00, 0.0, 0.0, 0.0],
                [1.0, 0.80, 1.00, 0.00, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),

        "edge_index": torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4],
                [1, 0, 2, 1, 3, 2, 4, 3],
            ],
            dtype=torch.long,
        ),

        "edge_features": torch.tensor(
            [
                [0.20, 100.0, 5.0, 0.01, 0.0],
                [0.20, 100.0, 5.0, 0.01, 0.0],
                [0.35, 100.0, 7.0, 0.02, 0.0],
                [0.35, 100.0, 7.0, 0.02, 0.0],
                [0.65, 100.0, 10.0, 0.03, 0.0],
                [0.65, 100.0, 10.0, 0.03, 0.0],
                [0.85, 100.0, 15.0, 0.05, 0.0],
                [0.85, 100.0, 15.0, 0.05, 0.0],
            ],
            dtype=torch.float32,
        ),

        "demand_features": torch.tensor(
            [[30.0]],
            dtype=torch.float32,
        ),
    }


def test_congestion_action():

    agent = CongestionAgent(
        batch_size=2,
        epsilon_start=0.0,
    )

    action = agent.select_action(
        make_state(),
        training=False,
    )

    assert action in [0, 1, 2]


def test_congestion_learning():

    agent = CongestionAgent(
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


def test_congestion_mode_restore():

    agent = CongestionAgent(
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