# Isolated live SDN routing demo

This directory adds an independently launchable OpenFlow 1.3 controller and a
Mininet test topology. It leaves `generic_controller.py` and the prior
`multipath_topology.py` intact.

## What actually makes decisions

`LiveRoutingController` discovers switches and directed LLDP links through Ryu,
learns host MAC/IP attachment points, computes a simple loop-free switch path,
and issues flow modifications for forward and reverse Ethernet pairs. Its
default policy is shortest-hop routing. Set `LIVE_CONGESTION_ROUTING=1` to use
measured link utilization as a tie-breaking/avoidance cost when every edge on
the topology has a fresh counter sample. A failed/disappeared link is excluded
from the discovered graph; restoration makes it eligible again. If no route
exists, the packet is not forwarded.

Current Congestion and Failure DQN checkpoints are **not loaded or represented
as controlling this testbed**. Their graph encoder accepts variable graph
sizes, but their learned outputs are global simulator interventions (reroute,
rate-limit, isolate), not a next-hop/path action. Their trained observation
also includes node load, normalized coordinates, delay, packet loss, and offered
demand. The live controller does not have these fields with the same defined
semantics. It does not fill those gaps with zeros/defaults and does not invent
online rewards. The existing Orchestrator is rule-based and depends on those
simulator states; it is not wired into this live controller.

The Phase 3 DQN files are
`results/phase3_train_congestion_seed20261021/congestion_dqn_final.pt` and
`results/phase3_train_failure_seed20261022/failure_dqn_final.pt`. They are
PyTorch training checkpoints with `online_gnn`, `online_network`, target
network, optimizer, epsilon, and `learn_steps` entries. The graph encoder
expects 7 node features and 5 edge features; each policy returns three global
actions. The supervised scorer at
`results/dataset_training_supervised_seed31415/best_validation_model.pt` uses
the project's 11 candidate-link features and next-hop scoring interface; it is
not substituted into the controller because live congestion/queue features
and the matching candidate schema are not established here.

Port utilization is computed only when `LIVE_LINK_CAPACITY_MBPS` is explicitly
configured and two nondecreasing OpenFlow TX byte samples exist. The demo
topology shapes every link to 100 Mbps with Mininet `TCLink`, so that value is
known for that setup. This is an estimate from interface counters and configured
capacity, not queue occupancy or packet loss. Link delay and per-link loss are
not measured by this controller. When capacity is absent, utilization stays
unknown and congestion-aware path selection falls back to shortest hop count.

## Prerequisites

Run inside the Linux/WSL environment where Mininet, OVS, Ryu, NetworkX, and the
repository Python dependencies are installed. Mininet requires root privileges
and a working OVS service. Install the repository Python requirements plus Ryu
and Mininet using the environment's package manager. Confirm `ryu-manager`,
`mn`, `ovs-vsctl`, and `ovs-ofctl` are available. WSL2 deployments may not
support Mininet network namespaces/OVS in every kernel configuration; a
controller process starting does not prove the virtual network is working.

## Two terminal demo

Terminal 1, from the repository root in WSL:

```bash
cd ~/sdn-marl
LIVE_LINK_CAPACITY_MBPS=100 LIVE_CONGESTION_ROUTING=1 \
  ryu-manager --observe-links --ofp-tcp-listen-port 6653 \
  ryu.topology.switches live_sdn.marl_controller
```

Terminal 2, from the repository root in WSL:

```bash
cd ~/sdn-marl
sudo -E python3 live_sdn/live_demo_topology.py
```

At the Mininet prompt:

```text
mininet> h1 ping -c 5 h2
mininet> h2 ping -c 5 h1
mininet> links
mininet> sh ovs-ofctl -O OpenFlow13 dump-flows s1
```

The topology has two routes `s1-s2-s4` and `s1-s3-s4`. Inspect controller logs
for discovered links, selected routes, and flow-mod issuance. To test rerouting,
disable one actual Mininet link (for example `link s1 s2 down`), wait for Ryu's
topology discovery to report link removal, then run `h1 ping -c 5 h2` again.
Restore it with `link s1 s2 up`. If OVS CLI syntax differs in the installed
Mininet version, use `sh ovs-vsctl show` and set the specific interface link
state with `ovs-ofctl`/`ip link` from Mininet's `sh` command; a controller-side
flag alone is not a real failure injection.

## Tests and limits

From repository root:

```bash
python -m pytest -q tests/test_live_sdn_adapter.py
python -m compileall -q live_sdn
```

The pure routing tests cover directed port mapping, alternate paths after a
removed link, unknown measurements, congestion-weighted route choice, and
no-route behavior. An actual switch handshake, host ping, flow dump, and
Mininet link-down recovery require the Linux Mininet/OVS/Ryu runtime and must be
reported separately from unit tests. This controller currently provides no
online MARL inference, no QoS guarantees, and no complete stale-flow purge on
link changes; its short idle timeout limits stale reactive state while new
packets trigger recomputation.
