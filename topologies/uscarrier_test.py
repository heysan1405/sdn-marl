import time
import subprocess
import sys
import networkx as nx
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel

GML_FILE = "InternetTopologyZoo/gml/UsCarrier.gml"
TOPO_NAME = "UsCarrier"
CONTROLLER_IP = "127.0.0.1"
CONTROLLER_PORT = 6653
# Node IDs for h1 and h2 (diameter endpoints: Fort Myers and Carlisle)
H1_NODE_ID = 40   # Fort Myers
H2_NODE_ID = 147  # Carlisle

def log(msg):
    print(msg, flush=True)

def test():
    setLogLevel("info")

    log(f"\n=== {TOPO_NAME} LARGE-SCALE SDN TEST ===")
    log(f"Loading {GML_FILE}...")

    graph = nx.read_gml(GML_FILE, label="id")
    nodes_list = list(graph.nodes())
    h1_label = graph.nodes[H1_NODE_ID].get("label", "?")
    h2_label = graph.nodes[H2_NODE_ID].get("label", "?")

    log(f"  Nodes: {graph.number_of_nodes()}")
    log(f"  Edges: {graph.number_of_edges()}")
    log(f"  h1 -> node {H1_NODE_ID} ({h1_label})")
    log(f"  h2 -> node {H2_NODE_ID} ({h2_label})")

    # Build shortest path for reference
    path = nx.shortest_path(graph.to_undirected(), H1_NODE_ID, H2_NODE_ID)
    log(f"  Expected path length: {len(path)} hops")
    path_labels = [graph.nodes[n].get("label", str(n)) for n in path]
    log(f"  Path: {' -> '.join(path_labels)}")

    # Build Mininet network
    log(f"\n[1] Building Mininet network ({graph.number_of_nodes()} switches)...")

    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
    )

    controller = net.addController(
        "c0",
        controller=RemoteController,
        ip=CONTROLLER_IP,
        port=CONTROLLER_PORT
    )

    # Create switches — one per topology node
    node_to_switch = {}
    for index, node in enumerate(nodes_list, start=1):
        sw = net.addSwitch(f"s{index}", protocols="OpenFlow13")
        node_to_switch[node] = sw

    h1_sw_idx = nodes_list.index(H1_NODE_ID) + 1
    h2_sw_idx = nodes_list.index(H2_NODE_ID) + 1

    # Add hosts
    h1 = net.addHost("h1", ip="10.0.0.1/24")
    h2 = net.addHost("h2", ip="10.0.0.2/24")
    net.addLink(h1, node_to_switch[H1_NODE_ID], cls=TCLink, bw=100, delay="1ms", use_tbf=False)
    net.addLink(h2, node_to_switch[H2_NODE_ID], cls=TCLink, bw=100, delay="1ms", use_tbf=False)

    log(f"  h1 connected to s{h1_sw_idx} ({h1_label})")
    log(f"  h2 connected to s{h2_sw_idx} ({h2_label})")

    # Add topology links (undirected edges, one physical link each)
    log(f"[2] Creating {graph.number_of_edges()} physical links...")
    for u, v in graph.to_undirected().edges():
        net.addLink(
            node_to_switch[u],
            node_to_switch[v],
            cls=TCLink,
            bw=100,
            delay="1ms",
            use_tbf=False
        )

    log("[3] Starting network...")
    net.start()

    log("[4] Waiting 15 seconds for Ryu LLDP discovery...")
    time.sleep(15)

    # Connectivity test
    log("\n==================================================")
    log("           PING TEST: h1 -> h2")
    log("==================================================")
    ping_out = h1.cmd("ping -c 5 10.0.0.2")
    log(ping_out)

    # Extract loss line
    for line in ping_out.splitlines():
        if "packet loss" in line:
            log(f"  RESULT: {line.strip()}")

    # Flow tables on first and last switch in path
    log("\n==================================================")
    log("         SPOT-CHECK OPENFLOW RULES")
    log("==================================================")
    for sw_id in [H1_NODE_ID, H2_NODE_ID]:
        sw_idx = nodes_list.index(sw_id) + 1
        sw_label = graph.nodes[sw_id].get("label", "?")
        log(f"\n--- s{sw_idx} ({sw_label}) ---")
        flows = subprocess.check_output(
            f"ovs-ofctl -O OpenFlow13 dump-flows s{sw_idx}",
            shell=True, text=True
        )
        for line in flows.splitlines():
            if "priority=10" in line or "priority=0" in line:
                log("  " + line.strip())

    # Quick iperf test
    log("\n==================================================")
    log("              THROUGHPUT TEST (iperf)")
    log("==================================================")
    h2.cmd("iperf -s > /tmp/iperf_srv.log 2>&1 &")
    time.sleep(1)
    iperf_out = h1.cmd("iperf -c 10.0.0.2 -t 5")
    log(iperf_out)
    h2.cmd("killall iperf 2>/dev/null")

    # Port stats on the direct switch of h1
    log("\n==================================================")
    log(f"         LIVE PORT STATS: s{h1_sw_idx} ({h1_label})")
    log("==================================================")
    port_stats = subprocess.check_output(
        f"ovs-ofctl -O OpenFlow13 dump-ports s{h1_sw_idx}",
        shell=True, text=True
    )
    log(port_stats)

    log("\nStopping network...")
    net.stop()

    log(f"\n=== {TOPO_NAME} TEST COMPLETE ===")

if __name__ == "__main__":
    test()
