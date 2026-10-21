# MARL Live Routing

This implementation is isolated in `marl_live/`; it does not alter the legacy
agents, the earlier `live_sdn/` demo, historical project outputs, or old
checkpoints. The detailed code audit and experiment report is
[PHASE_NEXT_REPORT.md](./PHASE_NEXT_REPORT.md).

## Current architecture

The policy scores a variable set of candidate next hops. Observation
`candidate-link-v2` has six ordered features: utilization, utilization-known,
log-normalized capacity, hop progress, failed flag, weighted-cost progress.
Failed edges, visited nodes, candidates that cannot reach the destination
without revisiting, and max-hop actions are masked in both training and
inference and rechecked by `RoutingEnv.step()`.

The congestion actor trains on normal/congestion episodes. Its per-hop reward
is `-(1 + 3 * background_utilization)`, plus delivery or dead-end reward, which
matches the weighted-Dijkstra baseline's edge cost. The failure actor trains
for delivery and hop efficiency under failure/combined scenarios. The
rule-based Orchestrator uses the congestion actor when no failure is observed;
under failure it combines both masked policy distributions with equal
log-probability weights. It is not a learned Orchestrator.

The environment generates an eight-switch three-path graph with capacities,
background loads, one-link failures, and cross-link density sampled from
0.15–0.45 during training. It reports background utilization along the selected
path; it does not simulate or measure packet throughput, delay, loss, queues,
or dynamic recovery time. It is not a Mininet result.

## Reproduce training and evaluation

Run from the repository root in a Python environment with PyTorch, NumPy, and
NetworkX. Each output directory must be new or empty; commands refuse to
overwrite a populated directory.

```bash
python -m marl_live.train --agent congestion --episodes 1000 --seed 20262220 --out marl_live/runs/reproduction/congestion_20262220
python -m marl_live.train --agent failure --episodes 1000 --seed 20262230 --out marl_live/runs/reproduction/failure_20262230
python -m marl_live.evaluate --congestion-policy marl_live/runs/v4_train_seed20262220/congestion/congestion_policy.pt --failure-policy marl_live/runs/v4_train_seed20262230/failure/failure_policy.pt --episodes 100 --seed 99000 --cross-link-probability 0.35 --out marl_live/runs/reproduction/test_99000
python -m marl_live.analyze --runs marl_live/runs/v4_test_20_30 marl_live/runs/v4_test_21_31 marl_live/runs/v4_test_22_32 --training-runs marl_live/runs/v4_train_seed20262220/congestion marl_live/runs/v4_train_seed20262221/congestion marl_live/runs/v4_train_seed20262222/congestion marl_live/runs/v4_train_seed20262230/failure marl_live/runs/v4_train_seed20262231/failure marl_live/runs/v4_train_seed20262232/failure --out marl_live/runs/reproduction/analysis --bootstrap 5000 --seed 607
```

Current finalized checkpoints and per-episode CSV logs are in
`runs/v4_train_seed20262220/`, `runs/v4_train_seed20262221/`,
`runs/v4_train_seed20262222/` for congestion and the corresponding
`v4_train_seed20262230/`–`v4_train_seed20262232/` directories for failure.
Each training run used 1,000 episodes and 1,000 optimizer updates. Validation
seeds 97100–97199 used cross-link probability 0.25. Final held-out test seeds
99000–99099 used probability 0.35. Training samples randomized that parameter
in [0.15, 0.45]. No test results were used for selection.

## Final test summary

Each condition has 300 episode results: 100 paired test seeds evaluated by
three trained checkpoint pairs. All policies delivered 300/300 in every
condition. The Orchestrator had zero invalid actions, fallbacks, loops, and dead
ends. Values below are mean background utilization on selected route links
(standard deviation):

| Condition | Shortest path | Weighted congestion baseline | MARL Orchestrator |
|---|---:|---:|---:|
| Normal | 0.155 (0.076) | 0.069 (0.031) | 0.073 (0.036) |
| Congestion | 0.702 (0.830) | 0.088 (0.042) | 0.089 (0.042) |
| Failure | 0.137 (0.070) | 0.075 (0.033) | 0.078 (0.036) |
| Combined | 0.574 (0.640) | 0.109 (0.056) | 0.122 (0.117) |

The non-RL weighted baseline remains slightly better than MARL. Paired MARL
minus baseline utilization differences (100 held-out seed units, percentile
bootstrap 95% CIs) were: normal +0.00423 [+0.00221, +0.00675], congestion
+0.00059 [-0.00112, +0.00192], failure +0.00364 [+0.00217, +0.00537], and
combined +0.01316 [+0.00371, +0.02494]. Confidence intervals are descriptive,
not multiplicity-adjusted. Raw runs and all ablations are under
`runs/v4_test_*/`; summaries and paired comparisons are in
`runs/v4_test_analysis_final/`.

Compared with the preserved v1 run on the already-inspected seed set 93000,
the Orchestrator improved combined delivery from 99% to 100% and reduced route
utilization from 0.837 to 0.117. Congestion delivery rose from 98.3% to 100%.
Those paired results are explicitly diagnostic because the historical test
seeds were reused; they are not the fresh-test claim. See
`runs/v4_before_after_reused93000/`.

## Dependencies and live demo

The inspected WSL environment has Mininet, OVS, and NetworkX, but no Ryu,
PyTorch, or Python 3.12. The Windows Python 3.13 environment used for training
and automated tests has PyTorch. Neither environment was modified. The [Ryu
PyPI release](https://pypi.org/project/ryu/) is old; an upstream issue points
to a community compatibility fork for Python 3.11/3.12 and documents the
upstream maintenance/compatibility status
([issue](https://github.com/faucetsdn/ryu/issues/211)). An isolated venv install
can be created without changing the existing Python environment (requires
Python 3.12 from the WSL distro/package source):

```bash
sudo apt install python3.12 python3.12-venv
python3.12 -m venv ~/venvs/sdn-marl-live
source ~/venvs/sdn-marl-live/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy networkx
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'ryu4312 @ git+https://github.com/olibrius/ryu4312.git@v4.35.1'
python -c 'import torch, ryu; print(torch.__version__, ryu.__file__)'
```

This was not run in WSL; the `ryu4312` compatibility fork is not the upstream
release. Mininet installation is documented by its
[maintainers](https://mininet.org/download/).

If dependencies are installed in an isolated WSL Python 3.12 venv, start the
controller from the repository root:

```bash
source ~/venvs/sdn-marl-live/bin/activate
MARL_LIVE_LINK_CAPACITY_MBPS=100 \
MARL_LIVE_CONGESTION_POLICY=marl_live/runs/v4_train_seed20262220/congestion/congestion_policy.pt \
MARL_LIVE_FAILURE_POLICY=marl_live/runs/v4_train_seed20262230/failure/failure_policy.pt \
ryu-manager --observe-links --ofp-tcp-listen-port 6653 \
  ryu.topology.switches marl_live.ryu_controller
```

In a second WSL terminal, run `sudo python3 marl_live/mininet_topology.py
--capacity-mbps 100`. In its Mininet prompt:

```text
mininet> h1 ping -c 5 h2
mininet> h1 iperf -s -u -p 5001
mininet> h2 iperf -c 10.0.0.1 -u -p 5001 -b 80M -t 30
mininet> sh ovs-ofctl -O OpenFlow13 dump-flows s1
mininet> link s1 s2 down
mininet> h1 ping -c 5 h2
mininet> link s1 s2 up
mininet> h1 ping -c 5 h2
```

These live instructions have not been run with the MARL controller. A Mininet
base `mn --test pingall` passed previously using a standalone OVS bridge, which
does not verify this controller or learned forwarding. Ryu/torch remain absent
in WSL, so live deployment is still blocked and not claimed.

## Tests

```bash
python -m pytest -q
python -m compileall -q marl_live
```

Latest full test run: 48 passed, 260 PyTorch Geometric deprecation warnings.
The tests cover observation consistency, reachability-safe masks, reward
alignment, failure coordination, route loops, checkpoint compatibility, and
the prior project suite.
