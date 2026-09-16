

# ============================================================
# Feature dimensions
# ============================================================

# Node features:
# [degree, node_load, normalized_x, normalized_y,
#  is_source, is_destination]
NODE_FEATURE_DIM = 6

# Edge features:
# [utilization, capacity, delay, packet_loss]
EDGE_FEATURE_DIM = 4

# Demand features:
# [traffic_volume]
DEMAND_FEATURE_DIM = 1

# Candidate path features:
# [hop_count, bottleneck_utilization, total_delay]
PATH_FEATURE_DIM = 3


# ============================================================
# GNN configuration
# ============================================================

# Hidden representation inside the GNN.
GNN_HIDDEN_DIM = 64

# Final graph representation size.
GRAPH_EMBEDDING_DIM = 128

# Number of GNN layers.
GNN_LAYERS = 2

# Number of attention heads in GAT.
GNN_HEADS = 4

# Dropout used by the GNN.
GNN_DROPOUT = 0.1


# ============================================================
# Routing network configuration
# ============================================================

# Hidden dimension of the path-scoring network.
ROUTING_HIDDEN_DIM = 128


# ============================================================
# DQN configuration
# ============================================================

# Adam optimizer learning rate.
LEARNING_RATE = 1e-3

# Discount factor for future rewards.
GAMMA = 0.99

# Initial exploration probability.
EPSILON_START = 1.0

# Minimum exploration probability.
EPSILON_END = 0.05

# Multiplicative epsilon decay after each learning step.
EPSILON_DECAY = 0.995


# ============================================================
# Replay buffer configuration
# ============================================================

# Maximum number of experiences stored.
REPLAY_CAPACITY = 100_000

# Number of experiences sampled for one learning step.
BATCH_SIZE = 32


# ============================================================
# Target network configuration
# ============================================================

# Copy online networks to target networks every N
# learning steps.
TARGET_UPDATE_FREQUENCY = 10


# ============================================================
# Training configuration
# ============================================================

# Total number of training episodes.
NUM_EPISODES = 500

# Maximum environment steps allowed in one episode.
#
# Person 1's environment should normally decide when an
# episode terminates. This value is only a safety limit.
MAX_STEPS_PER_EPISODE = 100

# Save a model checkpoint every N episodes.
CHECKPOINT_FREQUENCY = 50

# Print training information every N episodes.
LOG_FREQUENCY = 10

# Directory where training checkpoints are stored.
CHECKPOINT_DIR = "results/training"


# ============================================================
# Device configuration
# ============================================================

# Set to None to automatically use CUDA when available,
# otherwise CPU.
DEVICE = None