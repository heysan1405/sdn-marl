import sys
import os
import argparse
import time
import subprocess
import networkx as nx
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel


def load_gml_topology(gml_identifier):
    """Loads a GML file by path or name from InternetTopologyZoo/gml/"""
    if os.path.exists(gml_identifier):
        filepath = gml_identifier
    else:
        # Search in InternetTopologyZoo/gml/
        zoo_dir = "InternetTopologyZoo/gml"
        if not gml_identifier.endswith(".gml"):
            gml_identifier += ".gml"
        filepath = os.path.join(zoo_dir, gml_identifier)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GML topology file not found: {filepath}")

    graph = nx.read_gml(filepath, label="id")
    return filepath, graph


from mininet.cli import CLI


def run_test(gml_identifier, controller_ip="127.0.0.1", controller_port=6653, interactive=False, keep_alive=False, traffic=False):
    setLogLevel("info")

    filepath, graph = load_gml_topology(gml_identifier)
    name = os.path.splitext(os.path.basename(filepath))[0]

    print("\n" + "=" * 60)
    print(f"   UNIVERSAL TOPOLOGY RUNNER: {name}")
    print(f"   File: {filepath}")
    print(f"   Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")
    print("=" * 60)

    # Get largest connected component
    ug = graph.to_undirected()
    if not nx.is_connected(ug):
        comps = list(nx.connected_components(ug))
        largest = max(comps, key=len)
        print(f"[Warning] Graph has {len(comps)} components. Using largest component ({len(largest)} nodes).")
        graph = graph.subgraph(largest).copy()
        ug = graph.to_undirected()

    nodes_list = list(graph.nodes())

    # Find periphery endpoints for h1 and h2
    periphery = nx.periphery(ug)
    h1_node = periphery[0]
    h2_node = periphery[-1] if len(periphery) > 1 else nodes_list[-1]

    h1_label = graph.nodes[h1_node].get("label", str(h1_node))
    h2_label = graph.nodes[h2_node].get("label", str(h2_node))
    shortest_path = nx.shortest_path(ug, h1_node, h2_node)

    print(f"\n[Endpoints Selected]")
    print(f"  h1 -> node {h1_node} ({h1_label})")
    print(f"  h2 -> node {h2_node} ({h2_label})")
    print(f"  Shortest path ({len(shortest_path)} hops): {' -> '.join([str(graph.nodes[n].get('label', n)) for n in shortest_path])}")

    # Build Mininet Network
    print(f"\n[1] Building Mininet switches ({graph.number_of_nodes()} nodes)...")
    net = Mininet(controller=None, switch=OVSSwitch, link=TCLink, autoSetMacs=True)

    net.addController("c0", controller=RemoteController, ip=controller_ip, port=controller_port)

    # Add switches
    node_to_sw = {}
    for idx, node in enumerate(nodes_list, start=1):
        sw = net.addSwitch(f"s{idx}", protocols="OpenFlow13")
        node_to_sw[node] = sw

    # Add hosts
    h1_sw = node_to_sw[h1_node]
    h2_sw = node_to_sw[h2_node]
    h1 = net.addHost("h1", ip="10.0.0.1/24")
    h2 = net.addHost("h2", ip="10.0.0.2/24")
    net.addLink(h1, h1_sw, cls=TCLink, bw=100, delay="1ms", use_tbf=False)
    net.addLink(h2, h2_sw, cls=TCLink, bw=100, delay="1ms", use_tbf=False)

    # Add inter-switch links
    print(f"[2] Linking {graph.number_of_edges()} physical edges...")
    for u, v in ug.edges():
        net.addLink(node_to_sw[u], node_to_sw[v], cls=TCLink, bw=100, delay="1ms", use_tbf=False)

    print("[3] Starting network...")
    net.start()

    print("[4] Waiting 20s for complete LLDP topology discovery...")
    time.sleep(20)

    print("\n" + "=" * 50)
    print("           PING TEST: h1 -> h2")
    print("=" * 50)
    ping_out = h1.cmd("ping -c 5 10.0.0.2")
    print(ping_out)

    print("\n" + "=" * 50)
    print("           THROUGHPUT TEST (iperf)")
    print("=" * 50)
    h2.cmd("iperf -s > /tmp/iperf_srv.log 2>&1 &")
    time.sleep(1)
    iperf_out = h1.cmd("iperf -c 10.0.0.2 -t 5")
    print(iperf_out)

    if traffic:
        print("\n[Continuous Bottleneck Traffic Stream Injected: 85 Mbps h1 -> h2]")
        h1.cmd("iperf -c 10.0.0.2 -t 3600 -b 85M > /dev/null 2>&1 &")

    if interactive:
        print("\n[Entering Mininet Interactive CLI... Type 'exit' to stop]")
        CLI(net)
    elif keep_alive:
        print("\n[Network Keep-Alive Active] Continuous traffic running... Press Ctrl+C to stop network.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    h2.cmd("killall iperf 2>/dev/null")
    h1.cmd("killall iperf 2>/dev/null")
    print("\nStopping network...")
    net.stop()
    print(f"=== TEST FOR {name} COMPLETE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal SDN Topology Test Runner")
    parser.add_argument("--gml", type=str, required=True, help="Topology name (e.g. Abilene, UsCarrier, Dfn) or path to .gml file")
    parser.add_argument("--cli", action="store_true", help="Drop into interactive Mininet CLI after starting")
    parser.add_argument("--keep-alive", action="store_true", help="Keep network running continuously for live agent monitoring")
    parser.add_argument("--traffic", action="store_true", help="Generate continuous background traffic stream during keep-alive")
    args = parser.parse_args()

    run_test(args.gml, interactive=args.cli, keep_alive=args.keep_alive, traffic=args.traffic)
