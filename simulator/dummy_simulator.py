"""
Dummy SDN routing environment.

This environment is ONLY for testing the Person 2
GNN + DQN training pipeline.

It is intentionally simple and does not implement
the real Internet Topology Zoo simulator.

The environment follows the agreed state contract:

    node_features   [N, 6]
    edge_index      [2, E]
    edge_features   [E, 4]
    demand_features [1, 1]
    path_features   [1, K, 3]
    action_mask     [1, K]

The real simulator can replace this environment later.
"""

import random

import torch


class DummySDNRoutingEnvironment:
    """
    Small dummy environment for testing the routing agent.

    The environment creates a simple chain topology:

        0 -- 1 -- 2 -- 3 -- 4 -- ... -- N-1

    A source and destination are selected randomly.

    Several candidate paths are generated synthetically.

    The agent chooses one candidate path.

    A synthetic reward is then generated based on the
    candidate path's utilization and delay.
    """

    def __init__(
        self,
        min_nodes: int = 5,
        max_nodes: int = 8,
        num_candidate_paths: int = 4,
        max_steps: int = 10,
        seed: int | None = None,
    ):
        if min_nodes < 2:
            raise ValueError(
                "min_nodes must be at least 2."
            )

        if max_nodes < min_nodes:
            raise ValueError(
                "max_nodes must be greater than or equal "
                "to min_nodes."
            )

        if num_candidate_paths <= 0:
            raise ValueError(
                "num_candidate_paths must be greater than 0."
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than 0."
            )

        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.num_candidate_paths = num_candidate_paths
        self.max_steps = max_steps

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.num_nodes = None
        self.source = None
        self.destination = None

        self.current_step = 0

        self.node_features = None
        self.edge_index = None
        self.edge_features = None

        self.demand_features = None
        self.path_features = None
        self.action_mask = None

        self.candidate_paths = []

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):
        """
        Start a new episode.

        Returns:
            state dictionary.
        """

        self.current_step = 0

        # --------------------------------------------------
        # Generate a random topology size.
        # --------------------------------------------------

        self.num_nodes = random.randint(
            self.min_nodes,
            self.max_nodes,
        )

        # --------------------------------------------------
        # Select source and destination.
        # --------------------------------------------------

        self.source = 0

        self.destination = self.num_nodes - 1

        # --------------------------------------------------
        # Build topology.
        # --------------------------------------------------

        self._build_topology()

        # --------------------------------------------------
        # Generate traffic demand.
        # --------------------------------------------------

        self._generate_demand()

        # --------------------------------------------------
        # Generate candidate paths.
        # --------------------------------------------------

        self._generate_candidate_paths()

        return self._get_state()

    # ======================================================
    # Step
    # ======================================================

    def step(self, action):
        """
        Apply the selected routing action.

        Args:
            action:
                Integer candidate-path index.

        Returns:
            next_state,
            reward,
            done,
            info
        """

        if self.path_features is None:
            raise RuntimeError(
                "Environment has not been reset. "
                "Call reset() before step()."
            )

        # --------------------------------------------------
        # Validate action.
        # --------------------------------------------------

        if not isinstance(action, int):
            raise TypeError(
                "action must be an integer."
            )

        if action < 0:
            raise ValueError(
                "action cannot be negative."
            )

        if action >= self.num_candidate_paths:
            raise ValueError(
                f"Invalid action {action}. "
                f"Valid range: "
                f"0 to {self.num_candidate_paths - 1}."
            )

        if not self.action_mask[0, action]:
            raise ValueError(
                f"Action {action} is not valid "
                "according to action_mask."
            )

        # --------------------------------------------------
        # Calculate reward.
        # --------------------------------------------------

        reward = self._calculate_reward(
            action
        )

        # --------------------------------------------------
        # Update environment.
        # --------------------------------------------------

        self._apply_action_effect(
            action
        )

        self.current_step += 1

        # --------------------------------------------------
        # Determine episode termination.
        # --------------------------------------------------

        done = (
            self.current_step >= self.max_steps
        )

        # --------------------------------------------------
        # Generate next demand/path state.
        # --------------------------------------------------

        if not done:

            self._generate_candidate_paths()

        next_state = self._get_state()

        # --------------------------------------------------
        # Information for logging/evaluation.
        # --------------------------------------------------

        selected_path = self.candidate_paths[
            action
        ]

        info = {
            "step": self.current_step,
            "action": action,
            "selected_path": selected_path,
            "source": self.source,
            "destination": self.destination,
        }

        return (
            next_state,
            reward,
            done,
            info,
        )

    # ======================================================
    # Topology generation
    # ======================================================

    def _build_topology(self):
        """
        Build a simple chain topology.

        Example for 5 nodes:

            0 -- 1 -- 2 -- 3 -- 4

        Edges are stored in both directions so that the
        graph can be processed as an undirected network
        by the GNN.
        """

        edges = []

        for node in range(
            self.num_nodes - 1
        ):

            next_node = node + 1

            edges.append(
                (node, next_node)
            )

            edges.append(
                (next_node, node)
            )

        self.edge_index = torch.tensor(
            edges,
            dtype=torch.long,
        ).t().contiguous()

        # --------------------------------------------------
        # Edge features
        # --------------------------------------------------
        #
        # [utilization, capacity, delay, packet_loss]
        #

        num_edges = len(edges)

        edge_features = []

        for _ in range(num_edges):

            utilization = random.uniform(
                0.1,
                0.8,
            )

            capacity = 100.0

            delay = random.uniform(
                1.0,
                10.0,
            )

            packet_loss = random.uniform(
                0.0,
                0.05,
            )

            edge_features.append(
                [
                    utilization,
                    capacity,
                    delay,
                    packet_loss,
                ]
            )

        self.edge_features = torch.tensor(
            edge_features,
            dtype=torch.float32,
        )

        # --------------------------------------------------
        # Node features
        # --------------------------------------------------
        #
        # [degree, node_load, normalized_x,
        #  normalized_y, is_source, is_destination]
        #

        node_features = []

        for node in range(
            self.num_nodes
        ):

            if node == 0:

                degree = 1

            elif node == self.num_nodes - 1:

                degree = 1

            else:

                degree = 2

            node_load = random.uniform(
                0.1,
                0.8,
            )

            if self.num_nodes > 1:

                normalized_position = (
                    node
                    / (self.num_nodes - 1)
                )

            else:

                normalized_position = 0.0

            is_source = (
                1.0
                if node == self.source
                else 0.0
            )

            is_destination = (
                1.0
                if node == self.destination
                else 0.0
            )

            node_features.append(
                [
                    float(degree),
                    node_load,
                    normalized_position,
                    normalized_position,
                    is_source,
                    is_destination,
                ]
            )

        self.node_features = torch.tensor(
            node_features,
            dtype=torch.float32,
        )

    # ======================================================
    # Traffic generation
    # ======================================================

    def _generate_demand(self):
        """
        Generate a random traffic volume.
        """

        traffic_volume = random.uniform(
            10.0,
            50.0,
        )

        self.demand_features = torch.tensor(
            [[traffic_volume]],
            dtype=torch.float32,
        )

    # ======================================================
    # Candidate path generation
    # ======================================================

    def _generate_candidate_paths(self):
        """
        Generate synthetic candidate paths.

        The dummy topology is a chain, so there is only one
        physically meaningful path. To allow the RL agent
        to exercise its discrete action-selection mechanism,
        we create synthetic candidate paths with different
        QoS characteristics.

        This is NOT intended to represent real routing.
        Person 1's real routing implementation will replace
        this later.
        """

        self.candidate_paths = []

        for _ in range(
            self.num_candidate_paths
        ):

            # The actual chain path.
            path = list(
                range(
                    self.source,
                    self.destination + 1,
                )
            )

            self.candidate_paths.append(
                path
            )

        # --------------------------------------------------
        # Create candidate path features.
        # --------------------------------------------------

        path_features = []

        for path_index in range(
            self.num_candidate_paths
        ):

            hop_count = (
                len(self.candidate_paths[path_index])
                - 1
            )

            # Give each candidate a different synthetic
            # utilization/delay profile.

            bottleneck_utilization = random.uniform(
                0.2,
                0.95,
            )

            total_delay = random.uniform(
                5.0,
                30.0,
            )

            path_features.append(
                [
                    float(hop_count),
                    bottleneck_utilization,
                    total_delay,
                ]
            )

        self.path_features = torch.tensor(
            [path_features],
            dtype=torch.float32,
        )

        # --------------------------------------------------
        # Action mask.
        # --------------------------------------------------
        #
        # For the dummy environment, all candidate paths
        # are valid.
        #
        # The mask exists so the same training code can
        # later handle variable K and padded actions.
        #

        self.action_mask = torch.ones(
            (
                1,
                self.num_candidate_paths,
            ),
            dtype=torch.bool,
        )

    # ======================================================
    # Reward
    # ======================================================

    def _calculate_reward(
        self,
        action,
    ):
        """
        Calculate a synthetic routing reward.

        Lower utilization and lower delay produce a
        better reward.

        This is only for testing the RL pipeline.
        """

        utilization = float(
            self.path_features[
                0,
                action,
                1,
            ].item()
        )

        delay = float(
            self.path_features[
                0,
                action,
                2,
            ].item()
        )

        # Normalize delay approximately to [0, 1].
        normalized_delay = min(
            delay / 30.0,
            1.0,
        )

        # Synthetic reward:
        #
        # lower utilization -> better
        # lower delay       -> better

        reward = (
            1.0
            - utilization
            - normalized_delay
        )

        return float(reward)

    # ======================================================
    # Apply action
    # ======================================================

    def _apply_action_effect(
        self,
        action,
    ):
        """
        Apply a small synthetic effect to the network.

        This is intentionally simple.

        In the real environment, Person 1's network
        simulator will update link utilization, delay,
        packet loss, throughput, etc.
        """

        utilization = self.path_features[
            0,
            action,
            1,
        ].item()

        # Increase utilization on the selected path.
        increase = min(
            0.05,
            max(
                0.0,
                self.demand_features[
                    0,
                    0,
                ].item()
                / 1000.0,
            ),
        )

        new_utilization = min(
            1.0,
            utilization + increase,
        )

        self.path_features[
            0,
            action,
            1,
        ] = new_utilization

    # ======================================================
    # State
    # ======================================================

    def _get_state(self):
        """
        Return the current state in the exact format
        expected by RoutingAgent.
        """

        if self.node_features is None:
            raise RuntimeError(
                "Environment has not been initialized."
            )

        state = {
            "node_features": (
                self.node_features.clone()
            ),

            "edge_index": (
                self.edge_index.clone()
            ),

            "edge_features": (
                self.edge_features.clone()
            ),

            "demand_features": (
                self.demand_features.clone()
            ),

            "path_features": (
                self.path_features.clone()
            ),

            "action_mask": (
                self.action_mask.clone()
            ),
        }

        return state

    # ======================================================
    # Convenience methods
    # ======================================================

    @property
    def num_actions(self):
        """
        Number of candidate routing actions.
        """

        return self.num_candidate_paths

    @property
    def observation_space_shape(self):
        """
        Return the fixed feature dimensions.

        N, E and K are variable.
        """

        return {
            "node_features": (
                None,
                6,
            ),

            "edge_index": (
                2,
                None,
            ),

            "edge_features": (
                None,
                4,
            ),

            "demand_features": (
                1,
                1,
            ),

            "path_features": (
                1,
                None,
                3,
            ),

            "action_mask": (
                1,
                None,
            ),
        }