
import os
import tempfile

import torch

from agents import RoutingAgent


def create_mock_state(
    num_nodes=5,
    num_candidate_paths=4,
):
    """
    Create a small fake SDN routing state.

    This follows the agreed Person 1 -> Person 2
    state interface:

        node_features   [N, 6]
        edge_index      [2, E]
        edge_features   [E, 4]
        demand_features [1, 1]
        path_features   [1, K, 3]
        action_mask     [1, K]
    """

    # --------------------------------------------------
    # Node features
    # --------------------------------------------------
    #
    # [degree, node_load, normalized_x,
    #  normalized_y, is_source, is_destination]
    #

    node_features = torch.tensor(
        [
            [2.0, 0.20, 0.00, 0.00, 1.0, 0.0],
            [2.0, 0.30, 0.25, 0.25, 0.0, 0.0],
            [2.0, 0.40, 0.50, 0.50, 0.0, 0.0],
            [2.0, 0.50, 0.75, 0.75, 0.0, 0.0],
            [1.0, 0.20, 1.00, 1.00, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # --------------------------------------------------
    # Edge index
    # --------------------------------------------------

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3],
            [1, 0, 2, 1, 3, 2, 4],
        ],
        dtype=torch.long,
    )

    num_edges = edge_index.size(1)

    # --------------------------------------------------
    # Edge features
    # --------------------------------------------------
    #
    # [utilization, capacity, delay, packet_loss]
    #

    edge_features = torch.tensor(
        [
            [0.20, 100.0, 5.0, 0.00],
            [0.20, 100.0, 5.0, 0.00],
            [0.30, 100.0, 6.0, 0.01],
            [0.30, 100.0, 6.0, 0.01],
            [0.40, 100.0, 7.0, 0.01],
            [0.40, 100.0, 7.0, 0.01],
            [0.50, 100.0, 8.0, 0.02],
        ],
        dtype=torch.float32,
    )

    assert edge_features.size(0) == num_edges

    # --------------------------------------------------
    # Demand features
    # --------------------------------------------------
    #
    # [traffic_volume]
    #

    demand_features = torch.tensor(
        [[25.0]],
        dtype=torch.float32,
    )

    # --------------------------------------------------
    # Candidate path features
    # --------------------------------------------------
    #
    # [hop_count, bottleneck_utilization, total_delay]
    #

    path_features = torch.tensor(
        [
            [
                [2.0, 0.80, 10.0],
                [2.0, 0.40, 8.0],
                [3.0, 0.30, 12.0],
                [4.0, 0.90, 20.0],
            ]
        ],
        dtype=torch.float32,
    )

    assert path_features.size(1) == num_candidate_paths

    # --------------------------------------------------
    # Action mask
    # --------------------------------------------------
    #
    # True  = valid candidate path
    # False = invalid/padded candidate path
    #
    # In this test, the final path is intentionally
    # masked to make sure the agent never selects it.
    #

    action_mask = torch.tensor(
        [
            [True, True, True, False]
        ],
        dtype=torch.bool,
    )

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "demand_features": demand_features,
        "path_features": path_features,
        "action_mask": action_mask,
    }


def test_state_format():
    """
    Verify that the mock state follows the project contract.
    """

    state = create_mock_state()

    assert state["node_features"].shape == (5, 6)

    assert state["edge_index"].shape[0] == 2

    assert (
        state["edge_features"].shape[1] == 4
    )

    assert state["demand_features"].shape == (1, 1)

    assert state["path_features"].shape == (1, 4, 3)

    assert state["action_mask"].shape == (1, 4)

    assert state["action_mask"].dtype == torch.bool


def test_agent_select_action():
    """
    Verify that the agent can select a valid action.
    """

    agent = RoutingAgent(
        batch_size=4,
        replay_capacity=100,
    )

    state = create_mock_state()

    action = agent.select_action(
        state,
        training=False,
    )

    # Valid actions are 0, 1, and 2.
    assert action in [0, 1, 2]

    # The masked action 3 must never be selected.
    assert action != 3


def test_agent_exploration():
    """
    Verify epsilon-greedy exploration.

    With epsilon = 1.0, the agent should select only
    from valid actions, never from the masked action.
    """

    agent = RoutingAgent(
        epsilon_start=1.0,
        epsilon_end=1.0,
        batch_size=4,
        replay_capacity=100,
    )

    state = create_mock_state()

    for _ in range(50):

        action = agent.select_action(
            state,
            training=True,
        )

        assert action in [0, 1, 2]
        assert action != 3


def test_replay_buffer():
    """
    Verify that experiences can be stored and sampled.
    """

    agent = RoutingAgent(
        batch_size=4,
        replay_capacity=100,
    )

    state = create_mock_state()
    next_state = create_mock_state()

    for _ in range(4):

        agent.remember(
            state=state,
            action=0,
            reward=1.0,
            next_state=next_state,
            done=False,
        )

    assert len(agent.replay_buffer) == 4

    experiences = agent.replay_buffer.sample(4)

    assert len(experiences) == 4


def test_agent_learning():
    """
    Fill the replay buffer and verify that learning
    produces a finite loss.
    """

    agent = RoutingAgent(
        batch_size=4,
        replay_capacity=100,
        target_update_frequency=2,
    )

    state = create_mock_state()
    next_state = create_mock_state()

    # Add enough experiences for learning.
    for i in range(8):

        action = i % 3

        reward = float(i) * 0.1

        done = i == 7

        agent.remember(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    assert len(agent.replay_buffer) == 8

    initial_epsilon = agent.epsilon

    loss = agent.learn()

    assert loss is not None

    assert isinstance(loss, float)

    assert torch.isfinite(
        torch.tensor(loss)
    )

    # Epsilon should decrease after learning.
    assert agent.epsilon < initial_epsilon


def test_target_network_update():
    """
    Verify that the target networks can be updated from
    the online networks.
    """

    agent = RoutingAgent(
        batch_size=4,
        replay_capacity=100,
    )

    # Modify the online network slightly.
    with torch.no_grad():

        for parameter in agent.online_gnn.parameters():

            parameter.add_(0.001)

            break

    agent.update_target_network()

    online_parameters = list(
        agent.online_gnn.parameters()
    )

    target_parameters = list(
        agent.target_gnn.parameters()
    )

    assert len(online_parameters) == len(
        target_parameters
    )

    for online_parameter, target_parameter in zip(
        online_parameters,
        target_parameters,
    ):

        assert torch.equal(
            online_parameter,
            target_parameter,
        )


def test_save_and_load():
    """
    Verify that the agent can save and reload its
    neural networks and training state.
    """

    agent = RoutingAgent(
        batch_size=4,
        replay_capacity=100,
    )

    state = create_mock_state()

    # Run the network once so the model is exercised.
    action = agent.select_action(
        state,
        training=False,
    )

    assert action in [0, 1, 2]

    # Change training state so we can verify it is saved.
    agent.epsilon = 0.42
    agent.learn_steps = 17

    with tempfile.TemporaryDirectory() as temp_dir:

        checkpoint_path = os.path.join(
            temp_dir,
            "routing_agent.pt",
        )

        agent.save(checkpoint_path)

        assert os.path.exists(
            checkpoint_path
        )

        loaded_agent = RoutingAgent(
            batch_size=4,
            replay_capacity=100,
        )

        loaded_agent.load(
            checkpoint_path
        )

        assert loaded_agent.epsilon == 0.42

        assert loaded_agent.learn_steps == 17

        # Check online GNN parameters.
        for p1, p2 in zip(
            agent.online_gnn.parameters(),
            loaded_agent.online_gnn.parameters(),
        ):

            assert torch.equal(
                p1,
                p2,
            )

        # Check routing network parameters.
        for p1, p2 in zip(
            agent.online_routing.parameters(),
            loaded_agent.online_routing.parameters(),
        ):

            assert torch.equal(
                p1,
                p2
            )


def test_no_valid_next_actions():
    """
    Verify the DQN all-invalid-next-action edge case.

    If next_state contains no valid candidate paths,
    the bootstrap value should be zero rather than -inf.
    """

    agent = RoutingAgent(
        batch_size=1,
        replay_capacity=10,
        target_update_frequency=10,
    )

    state = create_mock_state()

    next_state = create_mock_state()

    # Make every next-state action invalid.
    next_state["action_mask"] = torch.tensor(
        [
            [False, False, False, False]
        ],
        dtype=torch.bool,
    )

    agent.remember(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state,
        done=False,
    )

    loss = agent.learn()

    assert loss is not None

    assert torch.isfinite(
        torch.tensor(loss)
    )


if __name__ == "__main__":
    # This allows:
    #
    #     python tests/test_agent.py
    #
    # to run the tests directly.
    #
    # pytest is still recommended.

    test_state_format()
    test_agent_select_action()
    test_agent_exploration()
    test_replay_buffer()
    test_agent_learning()
    test_target_network_update()
    test_save_and_load()
    test_no_valid_next_actions()

    print("All agent tests passed.")
