# MARL-Based Intelligent Routing in Software-Defined Networks (SDN)

> **Phase 1 & Phase 2 Baseline SDN + Mininet + Ryu Environment with Real-Time Network Monitoring**

This repository contains the baseline SDN routing infrastructure, dynamic topology loader, OpenFlow controller, and real-time monitoring module designed for Multi-Agent Reinforcement Learning (MARL) routing research.

---

## 🌟 Key Features

1. **Universal GML Topology Support**: Automatically loads and instantiates **any** topology from the [Internet Topology Zoo](http://www.topology-zoo.org/) (over 150 real-world networks including Abilene, UsCarrier, Dfn, GtsCe, Pern, etc.) without code modifications.
2. **Zero-Configuration Ryu Controller (`generic_controller.py`)**:
   - Dynamic OpenFlow 1.3 LLDP switch and link topology discovery.
   - Dynamic MAC and IP address learning on access ports (no hardcoded hosts).
   - Reactive flow rule installation with NetworkX shortest path fallback.
3. **Real-Time Network Monitoring (`monitoring/network_state.py`)**:
   - Periodically polls OpenFlow port statistics (`OFPPortStatsRequest`).
   - Computes real-time link throughput ($\text{Mbps}$), capacity utilization ($\%$), and packet loss rates.
   - Exposes a thread-safe, structured JSON state matrix designed as observations for MARL agents.
4. **Tested at Scale**: Validated on topologies ranging from 11 switches (Abilene) up to **158 switches & 189 links** (UsCarrier, 36-hop path).

---

## 📁 Repository Structure

```
sdn-marl/
├── README.md                   # Project documentation & execution guide
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── generic_controller.py       # Zero-configuration Ryu SDN Controller with monitoring
├── run_topology.py             # Universal Mininet Topology Loader & Test Runner
├── monitoring/
│   ├── __init__.py
│   └── network_state.py        # Real-time NetworkStateTracker class
├── topologies/
│   ├── abilene_controller.py   # Dedicated 11-node Abilene controller
│   ├── abilene_mininet.py      # Dedicated 11-node Abilene topology runner
│   ├── uscarrier_controller.py # Dedicated 158-node UsCarrier controller
│   └── uscarrier_test.py       # Dedicated 158-node UsCarrier topology runner
├── tests/
│   └── test_network_monitoring.py # Verification suite for monitoring & state polling
└── InternetTopologyZoo/gml/    # 150+ GML topology dataset files
```

---

## 🛠️ Prerequisites & Setup

- **OS**: Ubuntu Linux (or WSL2 Ubuntu on Windows)
- **Python**: Python 3.9 (recommended for Ryu compatibility)
- **Mininet**: Installed (`sudo apt install mininet`)
- **Open vSwitch (OVS)**: Installed (`sudo apt install openvswitch-switch`)

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd sdn-marl

# Install Python dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Universal Runner (Runs ANY Topology)

1. **Start the Generic Ryu Controller** (Terminal 1):
   ```bash
   ryu-manager --observe-links generic_controller.py --ofp-tcp-listen-port 6653
   ```

2. **Run any topology using `run_topology.py`** (Terminal 2):
   ```bash
   # Abilene (11 switches)
   sudo python3 run_topology.py --gml Abilene

   # Dfn (German Research Network - 58 switches)
   sudo python3 run_topology.py --gml Dfn

   # UsCarrier (158 switches, 189 links)
   sudo python3 run_topology.py --gml UsCarrier
   ```

---

### Option 2: Dedicated Test Scripts

```bash
# Run Abilene dedicated test
ryu-manager --observe-links topologies/abilene_controller.py &
sudo python3 topologies/abilene_mininet.py

# Run Network Monitoring Validation Suite
sudo python3 tests/test_network_monitoring.py
```

---

## 📊 Experimental Results

| Metric | Abilene (11 switches) | UsCarrier (158 switches) |
| :--- | :--- | :--- |
| **Switches / Links** | 11 switches / 14 links | 158 switches / 189 links |
| **Shortest Path** | 5 hops | 36 hops |
| **Ping Loss** | 0.0% | 0.0% |
| **Sustained Throughput (`iperf`)** | 92.5 Mbps | 87.8 Mbps |
| **Dynamic Topology Discovery** | 100% | 100% |
| **Real-time Monitoring** | Functional | Functional |

---

## 📄 License
MIT License
