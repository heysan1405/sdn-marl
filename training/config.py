"""
Training configuration for the two-agent SDN-MARL system.
"""

# =========================================================
# State dimensions
# =========================================================

NODE_FEATURE_DIM = 7
EDGE_FEATURE_DIM = 5
DEMAND_FEATURE_DIM = 1

# =========================================================
# GNN
# =========================================================

GNN_HIDDEN_DIM = 64
GRAPH_EMBEDDING_DIM = 128

GNN_LAYERS = 2
GNN_HEADS = 4
GNN_DROPOUT = 0.1

# =========================================================
# Agent decision networks
# =========================================================

AGENT_HIDDEN_DIM = 128

# =========================================================
# DQN
# =========================================================

LEARNING_RATE = 1e-3
GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# =========================================================
# Replay buffer
# =========================================================

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 32

# =========================================================
# Target network
# =========================================================

TARGET_UPDATE_FREQUENCY = 10

# =========================================================
# Training
# =========================================================

NUM_EPISODES = 50
MAX_STEPS_PER_EPISODE = 100

CHECKPOINT_FREQUENCY = 50
LOG_FREQUENCY = 10

CHECKPOINT_DIR = "results/training"

# =========================================================
# Device
# =========================================================

# Automatically use CUDA if available.
DEVICE = None