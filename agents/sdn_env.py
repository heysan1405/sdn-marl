import numpy as np
import time


class SDNCongestionEnv:
    """
    Gym-compatible Reinforcement Learning Environment for SDN Congestion Avoidance.

    State:
      - Link utilization vector (capacities normalized [0, 1])
      - Link packet loss rate vector
      - Global max utilization & average utilization

    Actions:
      - Discrete(num_links + 1):
        - 0: Baseline (uniform link weights = 1.0)
        - i (1 to num_links): Inflate cost of link (i-1) by multiplier (e.g., 5.0x) to reroute traffic away from bottleneck link.

    Reward:
      - Penalizes max link utilization, load variance, and packet loss.
      - Gives bonus for keeping max utilization below 60%.
    """

    def __init__(self, controller=None, sampling_interval=2.0):
        self.controller = controller
        self.sampling_interval = sampling_interval

        # Link list mapping for state vector consistency
        self.link_keys = []
        self._num_links = 0

        # State dimensions
        self.state_dim = 0
        self.action_dim = 0

    def set_controller(self, controller):
        self.controller = controller

    def _update_link_keys(self, structured_state):
        links_dict = structured_state.get("links", {})
        sorted_keys = sorted(links_dict.keys())
        if sorted_keys != self.link_keys:
            self.link_keys = sorted_keys
            self._num_links = len(self.link_keys)
            # State vector: [utilization_1..N, loss_1..N, max_util, mean_util]
            self.state_dim = (self._num_links * 2) + 2
            self.action_dim = self._num_links + 1

    def get_state_vector(self):
        if self.controller is None:
            # Fallback zero state for initialization
            dim = (self._num_links * 2) + 2 if self._num_links > 0 else 10
            return np.zeros(dim, dtype=np.float32)

        state_dict = self.controller.get_network_state()
        self._update_link_keys(state_dict)

        links_dict = state_dict.get("links", {})

        utils = []
        losses = []

        for key in self.link_keys:
            stats = links_dict.get(key, {})
            u = stats.get("utilization", 0.0)
            l = stats.get("packet_loss_pct", 0.0) / 100.0
            utils.append(float(u))
            losses.append(float(l))

        utils_arr = np.array(utils, dtype=np.float32)
        losses_arr = np.array(losses, dtype=np.float32)

        max_u = np.max(utils_arr) if len(utils_arr) > 0 else 0.0
        mean_u = np.mean(utils_arr) if len(utils_arr) > 0 else 0.0

        state_vector = np.concatenate([
            utils_arr,
            losses_arr,
            np.array([max_u, mean_u], dtype=np.float32)
        ])

        return state_vector

    def step(self, action):
        """
        Executes action, updates controller link weights, waits for sampling window,
        and computes reward & next state.
        """
        if self.controller is not None:
            weights = {}
            if action > 0 and (action - 1) < len(self.link_keys):
                congested_link_key = self.link_keys[action - 1]
                # Format: "s1_p1->s2_p2" or "(dpid_src, dpid_dst)"
                # Set heavy weight multiplier on selected congested link
                weights[congested_link_key] = 10.0

            # Update controller link weights
            self.controller.update_link_weights(weights)

        # Wait for environment to observe effects of rerouting
        time.sleep(self.sampling_interval)

        next_state = self.get_state_vector()
        reward = self.compute_reward(next_state)
        done = False
        info = {}

        return next_state, reward, done, info

    def reset(self):
        if self.controller is not None:
            self.controller.update_link_weights({})  # Reset to default weights
        time.sleep(1.0)
        return self.get_state_vector()

    def compute_reward(self, state_vector):
        if len(state_vector) < 2:
            return 0.0

        max_u = state_vector[-2]
        mean_u = state_vector[-1]

        # Extract packet losses
        num_links = (len(state_vector) - 2) // 2
        losses = state_vector[num_links:2 * num_links] if num_links > 0 else np.array([0.0])
        total_loss = np.sum(losses)

        # Reward formulation:
        # Negative penalty for high bottleneck utilization & load variance
        reward = - (3.0 * (max_u ** 2) + 1.0 * mean_u + 10.0 * total_loss)

        # Incentive bonus if bottleneck utilization is kept below threshold (60%)
        if max_u < 0.60:
            reward += 2.0
        elif max_u > 0.85:
            reward -= 5.0  # Severe penalty for near-saturation

        return float(reward)
