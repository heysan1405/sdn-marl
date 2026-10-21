# MARL Live Routing: design and implementation status

## Audited facts

- The repository already contains GNN DQNs, a rule-based simulator Orchestrator,
  a generic Ryu controller, and an earlier isolated live demo. Existing DQNs
  choose among three global interventions, not next hops. They are not
  compatible with this system and are not loaded here.
- Current checkout is branch `master` and contains pre-existing uncommitted
  project changes and historical outputs. This implementation is isolated in
  `marl_live/`; those changes are preserved.
- WSL reports Mininet, OVS, and NetworkX installed, but not Ryu or PyTorch.
  The Windows Python used for repository testing has PyTorch. Actual live
  controller/deployment testing therefore depends on installing Ryu and a
  PyTorch-compatible runtime in WSL.

## Proposed system (implemented incrementally)

| Concern | Decision |
|---|---|
| Demo topology | Eight-switch, three-path graph with two cross links and explicit 100 Mbps shaped links; configurable in the topology launcher. |
| Traffic | Seeded background path loads in the training environment; live counters are sampled from OpenFlow port-stat replies. Offered demand is not reduced. |
| Candidate observation | Schema `candidate-link-v2`: utilization, utilization-known bit, log-normalized capacity, hop progress, failed bit, weighted-cost progress. Shape `[candidate_count, 6]`. Progress is recomputed on the remaining failure-free graph. |
| Actions | Categorical choice among current neighbors. Failed, visited, unreachable, and max-hop candidates are masked in training and inference, then checked by the environment/route builder. |
| Congestion agent | Learned neighbor ranking. Per-hop reward is `-(1 + 3 * background_utilization)`, matching the weighted-Dijkstra baseline edge objective, plus delivery/dead-end terminal rewards. |
| Failure agent | Learned valid-neighbor ranking on failure/combined episodes. Reward prioritizes delivery and hop efficiency; failed links cannot be selected. |
| Coordination | On observed failure, the rule-based Orchestrator fuses both masked policies with equal log-probability weights; otherwise congestion policy acts. It is not learned. |
| RL method | Shared candidate scorer with categorical policy and value baseline, trained by episodic actor-critic/REINFORCE. Variable degree is handled by scoring each candidate row, not a fixed action index. |
| Training/evaluation | Three training seeds per agent, 1,000 episodes per seed. Training cross-link density varies in [0.15, 0.45]; validation/test seeds and graph densities are disjoint. Evaluation includes baselines and actor/coordination ablations. |
| Live mapping | Controller builds the same candidate rows from discovered links and fresh port counters, then uses policy inference; selected neighbor is translated to its discovered output port. |
| Baselines/metrics | Shortest-hop route, non-RL utilization-penalized route, failed-link-excluding shortest route; delivery, hop count, path utilization, decision count, failure recovery. Live delay/loss remain unavailable unless independently instrumented. |

## Measurement semantics

Training-environment link loads are generated simulation values and are labeled
as such. Live utilization exists only if link capacity is configured and the
controller receives two valid monotonic byte-counter samples. A failed link is
identified from OpenFlow port/topology events and removed from the selectable
graph. Live queue depth, one-way delay, packet loss, and offered application
demand are not currently measured by this implementation and must not be
filled with synthetic values.

The live policy uses the same candidate feature builder as training. Capacity
ratio is `log1p(capacity_mbps)/log1p(capacity_reference_mbps)` and is
zero/unknown if capacity is unavailable. Utilization missingness is explicit.
Hop and weighted-cost progress are computed after removing visited nodes and
failed links; unreachable candidates are masked. Weighted progress uses the
same edge costs as the congestion reward and baseline.

The reported congestion measure is mean generated background utilization on
the selected route. It is comparable under paired topology/traffic seeds but
is not post-admission or network-wide utilization, throughput, or QoS. See
[PHASE_NEXT_REPORT.md](./PHASE_NEXT_REPORT.md) for diagnosis, metrics, CIs, and
limitations.

## Completion gates

Implemented modules, actual training checkpoints, automated tests, and live
Mininet results are reported separately. A learned policy is considered to
control forwarding only after a freshly loaded checkpoint makes a logged choice
that is translated to a next-hop port and verified in Mininet/OVS traffic and
flow-table evidence. Ryu is absent in the inspected WSL environment, so this
gate is currently blocked.
