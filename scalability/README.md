# Large topology evaluation

The CAIDA Matrix Market input is a topology only. The importer reads its symmetric coordinate pattern header, preserves one-based original IDs, canonicalizes undirected edges, and counts duplicate coordinates, self-loops and connected components. Selection takes a fixed-size seeded breadth-first traversal from a seeded start node, then keeps the induced edges. Each artifact includes compact local IDs and both mapping directions.

The evaluator samples distinct ordered source/destination pairs with a fixed seed. It caches BFS distances for sampled destinations only and avoids all-pairs shortest paths. The graph-only reference and distance-decreasing policy both choose shortest paths on this unweighted graph. Every constrained forwarding hop strictly reduces distance to that specific destination.

| Policy | Run status | Metrics available |
|---|---|---|
| Shortest-path reference | Completed | Delivery, failures, loops/dead ends, hops |
| Distance-constrained forwarding | Completed | Delivery, failures, loops/dead ends, hops |
| Baseline checkpoint | Blocked | Requires compatible 11-field link features |
| Congestion-aware checkpoint | Blocked | Requires utilization, occupancy and queue data |
| Congestion-aware + distance constraint | Blocked | Same missing learned-policy features |

The simulator can generate synthetic demands on a NetworkX graph, but uses assumed default capacities (1 Gbps) and delay (1 ms), then computes simplified utilization/loss. It does not create the saved model's port/queue feature vectors. CAIDA itself contains no traffic, capacity, queue, utilization or packet-loss observations. We therefore do not use simulated values as CAIDA measurements or as training data. A learned model evaluation needs a compatible link-state measurement source and an explicit traffic/capacity scenario; any simulator-based result must be labeled synthetic.

Graph-only results support conclusions about parsing, connected subgraphs, forwarding delivery, loop safety and hop count. They cannot support congestion-reduction, utilization, throughput or loss claims.

## Sampled routing at larger scales

The evaluator accepts any connected subgraph size and a full-graph mode. The
full-graph path parses the source once, samples a fixed set of pairs, and runs
one destination BFS per pair; it never allocates an all-pairs matrix. Learned
agent policies are explicitly unsupported in this topology-only script because
CAIDA has no traffic, failures, or link-state observations.

Reproduced with seed `314`:

| Graph | Nodes | Edges | Pairs | Delivered (both policies) | Mean hops | Runtime | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAIDA induced | 1,000 | 5,931 | 500 | 500 | 3.034 | 2.88 s | 283 MiB |
| CAIDA induced | 2,000 | 14,366 | 500 | 500 | 3.072 | 3.26 s | 283 MiB |
| CAIDA induced | 5,000 | 40,105 | 300 | 300 | 3.470 | 3.81 s | 283 MiB |
| CAIDA induced | 10,000 | 85,055 | 200 | 200 | 3.695 | 4.87 s | 283 MiB |
| Full CAIDA | 190,914 | 607,610 | 20 | 20 | 6.200 | 7.56 s | 286 MiB |

Every evaluated graph was connected; both shortest-path and distance-decreasing
forwarding had zero loops and zero dead ends in these samples. The constrained
policy strictly decreases destination-specific shortest-path distance at each
hop, so it matches the unweighted shortest-path hop count here. These samples
do not characterize all source/destination pairs.

## Commands

```bash
python scalability/evaluate_large_topology.py --size 1000 --seed 314 --pairs 500 --output-dir results/priority_scale
python scalability/evaluate_large_topology.py --size 2000 --seed 314 --pairs 500 --output-dir results/priority_scale
python scalability/evaluate_large_topology.py --size 5000 --seed 314 --pairs 300 --output-dir results/priority_scale
python scalability/evaluate_large_topology.py --size 10000 --seed 314 --pairs 200 --output-dir results/priority_scale
python scalability/evaluate_large_topology.py --full-graph --seed 314 --pairs 20 --output-dir results/priority_scale
pytest -q tests/test_large_topology.py
```

