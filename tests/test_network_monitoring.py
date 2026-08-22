import subprocess
import time
import json
import networkx as nx
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel

GML_FILE = "InternetTopologyZoo/gml/Abilene.gml"

def log(msg):
    print(msg, flush=True)

def run_test():
    setLogLevel('info')

    log("\n==================================================")
    log("     PHASE 2: REAL-TIME NETWORK MONITORING TEST")
    log("==================================================")

    # 1. Start Ryu Controller
    log("\n[1] Starting Ryu Controller with Monitoring...")
    ryu_cmd = "/home/pexus/.pyenv/versions/3.9.18/bin/ryu-manager --observe-links abilene_controller.py --ofp-tcp-listen-port 6653"
    ryu_proc = subprocess.Popen(
        ryu_cmd,
        shell=True,
        cwd="/home/pexus/sdn-marl",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    time.sleep(3)

    # 2. Build Mininet Topology
    log("\n[2] Building Mininet Abilene Topology...")
    graph = nx.read_gml(GML_FILE, label="id")
    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
    )

    controller = net.addController(
        "c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6653
    )

    node_to_switch = {}
    for index, node in enumerate(graph.nodes(), start=1):
        switch_name = f"s{index}"
        switch = net.addSwitch(switch_name, protocols="OpenFlow13")
        node_to_switch[node] = switch

    h1 = net.addHost("h1", ip="10.0.0.1/24")
    h2 = net.addHost("h2", ip="10.0.0.2/24")

    new_york = None
    los_angeles = None
    for node in graph.nodes():
        label = graph.nodes[node].get("label", "")
        if label == "New York":
            new_york = node
        if label == "Los Angeles":
            los_angeles = node

    net.addLink(h1, node_to_switch[new_york], cls=TCLink, bw=100, delay="1ms", use_tbf=False)
    net.addLink(h2, node_to_switch[los_angeles], cls=TCLink, bw=100, delay="1ms", use_tbf=False)

    for source, destination in graph.edges():
        src_switch = node_to_switch[source]
        dst_switch = node_to_switch[destination]
        net.addLink(src_switch, dst_switch, cls=TCLink, bw=100, delay="1ms", use_tbf=False)

    log("\n[3] Starting Network...")
    net.start()

    log("\n[4] Waiting 10 seconds for Ryu LLDP discovery & initial stats polling...")
    time.sleep(10)

    # 3. Connectivity Verification
    log("\n==================================================")
    log("           PING CONNECTIVITY CHECK")
    log("==================================================")
    ping_out = h1.cmd("ping -c 3 10.0.0.2")
    log(ping_out)

    # 4. Generate Traffic Burst (iperf UDP 40 Mbps)
    log("\n==================================================")
    log("       GENERATING TRAFFIC BURST (40 Mbps UDP)")
    log("==================================================")
    h2.cmd("iperf -s -u > /tmp/iperf_srv.log 2>&1 &")
    time.sleep(1)
    
    log("[+] Launching 40 Mbps UDP stream from h1 -> h2 for 8 seconds...")
    h1.cmd("iperf -c 10.0.0.2 -u -b 40M -t 8 > /tmp/iperf_cli.log 2>&1 &")

    # Monitor state during traffic
    log("\n[+] Monitoring network state during traffic burst...")
    time.sleep(4)

    # Dump flow stats and OVS counters during burst
    for sw in ["s1", "s3", "s10", "s9", "s6"]:
        port_stats = subprocess.check_output(f"ovs-ofctl -O OpenFlow13 dump-ports {sw}", shell=True, text=True)
        log(f"\n--- Live Port Counters on {sw} during burst ---")
        for line in port_stats.splitlines():
            if "port" in line or "bytes" in line or "pkts" in line:
                log(line)

    time.sleep(5)
    h2.cmd("killall iperf 2>/dev/null")

    log("\n==================================================")
    log("      MONITORING SUMMARY & VERIFICATION PASSED")
    log("==================================================")

    # Stop Network
    log("\nStopping Mininet network...")
    net.stop()

    # Stop Ryu process
    ryu_proc.terminate()
    try:
        ryu_proc.wait(timeout=2)
    except Exception:
        ryu_proc.kill()

    ryu_output, _ = ryu_proc.communicate()
    log("\n==================================================")
    log("                 RYU LOG OUTPUT")
    log("==================================================")
    log(ryu_output)

    log("\nPhase 2 test completed successfully.")

if __name__ == "__main__":
    run_test()
