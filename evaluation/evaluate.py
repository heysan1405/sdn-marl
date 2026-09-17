"""
Evaluation pipeline for SDN-MARL.

Methods:
    1. Trained two-agent MARL
    2. Shortest-path baseline

Supports:
    - seen topology evaluation
    - unseen topology evaluation
    - multiple evaluation episodes
    - CSV result generation
"""

import csv
import os
import random

import numpy as np
import torch

from agents.congestion_agent import CongestionAgent
from agents.failure_agent import FailureAgent

from evaluation.baseline import (
    run_shortest_path_baseline,
)

from evaluation.metrics import (
    extract_metrics,
)

from simulator.environment import SDNEnvironment

from training.config import (
    AGENT_HIDDEN_DIM,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    EDGE_FEATURE_DIM,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    GNN_DROPOUT,
    GNN_HEADS,
    GNN_HIDDEN_DIM,
    GNN_LAYERS,
    GRAPH_EMBEDDING_DIM,
    LEARNING_RATE,
    MAX_STEPS_PER_EPISODE,
    REPLAY_CAPACITY,
    TARGET_UPDATE_FREQUENCY,
)


# ============================================================
# Evaluation configuration
# ============================================================

TOPOLOGY_DIR = os.path.join(
    "InternetTopologyZoo",
    "gml",
)

TRAFFIC_SCENARIO = "normal"

NUM_DEMANDS = 10

K_PATHS = 5

NUM_EVALUATION_EPISODES = 10

SEED = 100


# ============================================================
# Checkpoint configuration
# ============================================================

# ------------------------------------------------------------
# TEMPORARY:
# The current training results contain episode-50 checkpoints.
#
# After training finishes and *_final.pt files exist,
# change this to:
#
#     CHECKPOINT_NAME = "final"
# ------------------------------------------------------------

CHECKPOINT_NAME = "episode_50"


CONGESTION_MODEL = os.path.join(
    CHECKPOINT_DIR,
    f"congestion_{CHECKPOINT_NAME}.pt",
)

FAILURE_MODEL = os.path.join(
    CHECKPOINT_DIR,
    f"failure_{CHECKPOINT_NAME}.pt",
)


# ============================================================
# Topology configuration
# ============================================================

SEEN_TOPOLOGIES = [
    "Abilene.gml",
]


# Geant2012.gml exists in the topology directory and was not
# used in the current training configuration.
UNSEEN_TOPOLOGIES = [
    "Geant2012.gml",
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    """Set random seeds."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# State validation
# ============================================================

def validate_state(state):
    """Validate the SDNEnvironment state contract."""

    required_keys = {
        "node_features",
        "edge_index",
        "edge_features",
        "demand_features",
    }

    missing = (
        required_keys
        - set(state.keys())
    )

    if missing:
        raise KeyError(
            f"Missing state keys: "
            f"{sorted(missing)}"
        )

    node_features = torch.as_tensor(
        state["node_features"]
    )

    edge_index = torch.as_tensor(
        state["edge_index"]
    )

    edge_features = torch.as_tensor(
        state["edge_features"]
    )

    demand_features = torch.as_tensor(
        state["demand_features"]
    )

    if node_features.dim() != 2:
        raise ValueError(
            "node_features must be "
            "2-dimensional."
        )

    if node_features.shape[1] != 7:
        raise ValueError(
            "node_features must have shape "
            "[N, 7]. "
            f"Got {tuple(node_features.shape)}."
        )

    if edge_index.dim() != 2:
        raise ValueError(
            "edge_index must be "
            "2-dimensional."
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape "
            "[2, E]."
        )

    if edge_features.dim() != 2:
        raise ValueError(
            "edge_features must be "
            "2-dimensional."
        )

    if edge_features.shape[1] != EDGE_FEATURE_DIM:
        raise ValueError(
            "edge_features must have shape "
            "[E, 5]. "
            f"Got {tuple(edge_features.shape)}."
        )

    if (
        edge_features.shape[0]
        != edge_index.shape[1]
    ):
        raise ValueError(
            "Number of edge features must "
            "match edge_index."
        )

    if demand_features.dim() != 2:
        raise ValueError(
            "demand_features must be "
            "2-dimensional."
        )

    if demand_features.shape != (1, 1):
        raise ValueError(
            "demand_features must have shape "
            "[1, 1]. "
            f"Got {tuple(demand_features.shape)}."
        )

    return True


# ============================================================
# Agent creation
# ============================================================

def create_agents():
    """Create congestion and failure agents."""

    congestion_agent = CongestionAgent(
        node_feature_dim=7,
        edge_feature_dim=EDGE_FEATURE_DIM,
        demand_feature_dim=1,

        gnn_hidden_dim=GNN_HIDDEN_DIM,
        graph_embedding_dim=GRAPH_EMBEDDING_DIM,
        gnn_layers=GNN_LAYERS,
        gnn_heads=GNN_HEADS,
        gnn_dropout=GNN_DROPOUT,

        hidden_dim=AGENT_HIDDEN_DIM,

        learning_rate=LEARNING_RATE,
        gamma=GAMMA,

        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,

        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,

        target_update_frequency=TARGET_UPDATE_FREQUENCY,

        device=DEVICE,
    )

    failure_agent = FailureAgent(
        node_feature_dim=7,
        edge_feature_dim=EDGE_FEATURE_DIM,
        demand_feature_dim=1,

        gnn_hidden_dim=GNN_HIDDEN_DIM,
        graph_embedding_dim=GRAPH_EMBEDDING_DIM,
        gnn_layers=GNN_LAYERS,
        gnn_heads=GNN_HEADS,
        gnn_dropout=GNN_DROPOUT,

        hidden_dim=AGENT_HIDDEN_DIM,

        learning_rate=LEARNING_RATE,
        gamma=GAMMA,

        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,

        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,

        target_update_frequency=TARGET_UPDATE_FREQUENCY,

        device=DEVICE,
    )

    return (
        congestion_agent,
        failure_agent,
    )


# ============================================================
# Load trained models
# ============================================================

def load_agents():
    """Create agents and load trained checkpoints."""

    print()
    print("Checkpoint configuration:")
    print(
        f"  Congestion: {CONGESTION_MODEL}"
    )
    print(
        f"  Failure:   {FAILURE_MODEL}"
    )

    if not os.path.isfile(
        CONGESTION_MODEL
    ):
        raise FileNotFoundError(
            "Congestion model not found: "
            f"{CONGESTION_MODEL}"
        )

    if not os.path.isfile(
        FAILURE_MODEL
    ):
        raise FileNotFoundError(
            "Failure model not found: "
            f"{FAILURE_MODEL}"
        )

    (
        congestion_agent,
        failure_agent,
    ) = create_agents()

    congestion_agent.load(
        CONGESTION_MODEL
    )

    failure_agent.load(
        FAILURE_MODEL
    )

    # Greedy evaluation.
    congestion_agent.epsilon = 0.0
    failure_agent.epsilon = 0.0

    congestion_agent.online_gnn.eval()
    congestion_agent.online_network.eval()

    failure_agent.online_gnn.eval()
    failure_agent.online_network.eval()

    return (
        congestion_agent,
        failure_agent,
    )


# ============================================================
# MARL evaluation
# ============================================================

def evaluate_marl_topology(
    topology_name,
    congestion_agent,
    failure_agent,
    num_episodes=10,
):
    """Evaluate trained MARL agents on one topology."""

    topology_path = os.path.join(
        TOPOLOGY_DIR,
        topology_name,
    )

    if not os.path.isfile(
        topology_path
    ):
        raise FileNotFoundError(
            f"Topology file not found: "
            f"{topology_path}"
        )

    results = []

    for episode in range(
        1,
        num_episodes + 1,
    ):

        seed = SEED + episode

        set_seed(seed)

        env = SDNEnvironment(
            topology_path=topology_path,
            traffic_scenario=TRAFFIC_SCENARIO,
            num_demands=NUM_DEMANDS,
            max_steps=MAX_STEPS_PER_EPISODE,
            k_paths=K_PATHS,
            random_seed=seed,
        )

        try:

            state = env.reset()

            validate_state(state)

            done = False
            steps = 0
            total_reward = 0.0

            last_info = {}

            while (
                not done
                and steps < MAX_STEPS_PER_EPISODE
            ):

                validate_state(state)

                congestion_action = (
                    congestion_agent.select_action(
                        state,
                        training=False,
                    )
                )

                failure_action = (
                    failure_agent.select_action(
                        state,
                        training=False,
                    )
                )

                (
                    next_state,
                    reward,
                    done,
                    info,
                ) = env.step(
                    congestion_action,
                    failure_action,
                )

                validate_state(
                    next_state
                )

                state = next_state

                total_reward += float(
                    reward
                )

                if isinstance(
                    info,
                    dict,
                ):
                    last_info = info
                else:
                    last_info = {}

                steps += 1

            # Metrics are returned under
            # info["metrics"] by SDNEnvironment.
            metrics = extract_metrics(
                last_info.get(
                    "metrics",
                    last_info,
                )
            )

            # Fallback to simulator metrics if
            # the info dictionary did not contain
            # usable metric values.
            if not metrics_have_values(
                metrics
            ):

                simulator = getattr(
                    env,
                    "simulator",
                    None,
                )

                if simulator is not None:

                    try:

                        metrics = extract_metrics(
                            simulator.calculate_network_metrics()
                        )

                    except Exception:
                        pass

            metrics.update(
                {
                    "method": "marl",
                    "topology": topology_name,
                    "episode": episode,
                    "steps": steps,
                    "total_reward": total_reward,
                }
            )

            results.append(metrics)

        finally:

            env.close()

    return results


def metrics_have_values(metrics):
    """
    Determine whether extracted metrics contain
    meaningful non-default values.
    """

    keys = [
        "max_utilization",
        "average_utilization",
        "delay",
        "packet_loss",
        "throughput",
        "congested_links",
    ]

    return any(
        float(
            metrics.get(
                key,
                0.0,
            )
        ) != 0.0
        for key in keys
    )


# ============================================================
# Baseline evaluation
# ============================================================

def evaluate_baseline_topology(
    topology_name,
    num_episodes=10,
):
    """Evaluate shortest-path baseline."""

    topology_path = os.path.join(
        TOPOLOGY_DIR,
        topology_name,
    )

    if not os.path.isfile(
        topology_path
    ):
        raise FileNotFoundError(
            f"Topology file not found: "
            f"{topology_path}"
        )

    results = []

    for episode in range(
        1,
        num_episodes + 1,
    ):

        seed = SEED + episode

        set_seed(seed)

        metrics = (
            run_shortest_path_baseline(
                topology_path=topology_path,
                traffic_scenario=TRAFFIC_SCENARIO,
                num_demands=NUM_DEMANDS,
                k_paths=K_PATHS,
                random_seed=seed,
            )
        )

        metrics["method"] = (
            "shortest_path"
        )

        metrics["topology"] = (
            topology_name
        )

        metrics["episode"] = (
            episode
        )

        results.append(metrics)

    return results


# ============================================================
# CSV saving
# ============================================================

def save_results(
    results,
    filepath,
):
    """Save evaluation results to CSV."""

    directory = os.path.dirname(
        filepath
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    fields = [
        "method",
        "topology",
        "episode",
        "steps",
        "total_reward",
        "max_utilization",
        "average_utilization",
        "delay",
        "packet_loss",
        "throughput",
        "congested_links",
    ]

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                result
            )


# ============================================================
# Summary
# ============================================================

def summarize_results(results):
    """Print mean results grouped by method/topology."""

    groups = {}

    for result in results:

        key = (
            result["method"],
            result["topology"],
        )

        groups.setdefault(
            key,
            [],
        ).append(result)

    print()
    print("=" * 90)
    print("EVALUATION SUMMARY")
    print("=" * 90)

    for (
        method,
        topology,
    ), rows in groups.items():

        print()
        print(
            f"Method:   {method}"
        )

        print(
            f"Topology: {topology}"
        )

        metrics = [
            "max_utilization",
            "average_utilization",
            "delay",
            "packet_loss",
            "throughput",
            "congested_links",
        ]

        for metric in metrics:

            values = []

            for row in rows:

                value = row.get(
                    metric,
                    None,
                )

                if value is not None:

                    try:

                        values.append(
                            float(value)
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

            if values:

                mean_value = (
                    sum(values)
                    / len(values)
                )

                print(
                    f"{metric:25s}: "
                    f"{mean_value:.6f}"
                )

    print()
    print("=" * 90)


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    output_directory = os.path.join(
        "results",
        "evaluation",
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    all_results = []

    print("=" * 90)
    print("SDN-MARL EVALUATION")
    print("=" * 90)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Evaluation episodes: "
        f"{NUM_EVALUATION_EPISODES}"
    )

    print(
        f"Checkpoint directory: "
        f"{CHECKPOINT_DIR}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_NAME}"
    )

    print(
        f"Congestion model: "
        f"{CONGESTION_MODEL}"
    )

    print(
        f"Failure model: "
        f"{FAILURE_MODEL}"
    )

    print("=" * 90)

    # ========================================================
    # Load trained agents
    # ========================================================

    print()
    print("Loading trained agents...")

    (
        congestion_agent,
        failure_agent,
    ) = load_agents()

    print(
        "Trained models loaded successfully."
    )

    # ========================================================
    # Seen topology evaluation
    # ========================================================

    print()
    print("=" * 90)
    print("SEEN TOPOLOGY EVALUATION")
    print("=" * 90)

    for topology_name in SEEN_TOPOLOGIES:

        print()
        print(
            f"Evaluating MARL on "
            f"{topology_name}..."
        )

        marl_results = (
            evaluate_marl_topology(
                topology_name=topology_name,
                congestion_agent=congestion_agent,
                failure_agent=failure_agent,
                num_episodes=NUM_EVALUATION_EPISODES,
            )
        )

        all_results.extend(
            marl_results
        )

        print(
            f"Evaluating shortest-path "
            f"baseline on {topology_name}..."
        )

        baseline_results = (
            evaluate_baseline_topology(
                topology_name=topology_name,
                num_episodes=NUM_EVALUATION_EPISODES,
            )
        )

        all_results.extend(
            baseline_results
        )

    # ========================================================
    # Unseen topology evaluation
    # ========================================================

    print()
    print("=" * 90)
    print("UNSEEN TOPOLOGY EVALUATION")
    print("=" * 90)

    for topology_name in UNSEEN_TOPOLOGIES:

        topology_path = os.path.join(
            TOPOLOGY_DIR,
            topology_name,
        )

        if not os.path.isfile(
            topology_path
        ):

            print()
            print(
                "WARNING: Unseen topology does "
                "not exist:"
            )

            print(
                f"  {topology_path}"
            )

            print(
                "Skipping this topology."
            )

            continue

        print()
        print(
            f"Evaluating MARL on unseen "
            f"topology: {topology_name}..."
        )

        marl_results = (
            evaluate_marl_topology(
                topology_name=topology_name,
                congestion_agent=congestion_agent,
                failure_agent=failure_agent,
                num_episodes=NUM_EVALUATION_EPISODES,
            )
        )

        all_results.extend(
            marl_results
        )

        print(
            f"Evaluating shortest-path "
            f"baseline on unseen "
            f"topology: {topology_name}..."
        )

        baseline_results = (
            evaluate_baseline_topology(
                topology_name=topology_name,
                num_episodes=NUM_EVALUATION_EPISODES,
            )
        )

        all_results.extend(
            baseline_results
        )

    # ========================================================
    # Save results
    # ========================================================

    results_path = os.path.join(
        output_directory,
        "evaluation_results.csv",
    )

    save_results(
        all_results,
        results_path,
    )

    # ========================================================
    # Summary
    # ========================================================

    summarize_results(
        all_results
    )

    print()
    print(
        f"Results saved to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()