import networkx as nx

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info


GML_FILE = "InternetTopologyZoo/gml/Abilene.gml"


# ============================================================
# ABILENE TOPOLOGY
# ============================================================

def load_abilene():

    graph = nx.read_gml(
        GML_FILE,
        label="id"
    )

    print("\n================================")
    print("       ABILENE TOPOLOGY")
    print("================================")

    print(f"Nodes : {graph.number_of_nodes()}")
    print(f"Edges : {graph.number_of_edges()}")

    return graph


# ============================================================
# CREATE MININET
# ============================================================

def create_network():

    graph = load_abilene()

    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
    )

    # --------------------------------------------------------
    # Controller
    # --------------------------------------------------------

    info("*** Adding remote controller\n")

    controller = net.addController(
        "c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6653
    )

    # --------------------------------------------------------
    # Create switches
    # --------------------------------------------------------

    info("*** Creating switches\n")

    node_to_switch = {}

    for index, node in enumerate(graph.nodes(), start=1):

        switch_name = f"s{index}"

        switch = net.addSwitch(
            switch_name,
            protocols="OpenFlow13"
        )

        node_to_switch[node] = switch

        info(
            f"  {switch_name} -> "
            f"{graph.nodes[node].get('label', node)}\n"
        )

    # --------------------------------------------------------
    # Hosts
    # --------------------------------------------------------

    info("*** Creating hosts\n")

    h1 = net.addHost(
        "h1",
        ip="10.0.0.1/24"
    )

    h2 = net.addHost(
        "h2",
        ip="10.0.0.2/24"
    )

    # --------------------------------------------------------
    # Choose endpoints
    #
    # Abilene:
    # New York      -> h1
    # Los Angeles   -> h2
    # --------------------------------------------------------

    new_york = None
    los_angeles = None

    for node in graph.nodes():

        label = graph.nodes[node].get(
            "label",
            ""
        )

        if label == "New York":
            new_york = node

        if label == "Los Angeles":
            los_angeles = node

    if new_york is None:
        raise Exception(
            "New York not found in Abilene topology"
        )

    if los_angeles is None:
        raise Exception(
            "Los Angeles not found in Abilene topology"
        )

    # --------------------------------------------------------
    # Connect hosts
    # --------------------------------------------------------

    info(
        f"\nHosts:\n"
        f"h1 -> New York\n"
        f"h2 -> Los Angeles\n\n"
    )

    net.addLink(
        h1,
        node_to_switch[new_york],
        cls=TCLink,
        bw=100,
        delay="1ms",
        use_tbf=False
    )

    net.addLink(
        h2,
        node_to_switch[los_angeles],
        cls=TCLink,
        bw=100,
        delay="1ms",
        use_tbf=False
    )

    # --------------------------------------------------------
    # Connect Abilene links
    # --------------------------------------------------------

    info("*** Creating Abilene links\n")

    for source, destination in graph.edges():

        src_switch = node_to_switch[source]
        dst_switch = node_to_switch[destination]

        # GML is treated as an undirected physical topology.
        #
        # Add only one Mininet link per physical connection.
        #
        # NetworkX read_gml may produce an undirected graph,
        # so graph.edges() gives each physical link once.

        net.addLink(
            src_switch,
            dst_switch,
            cls=TCLink,
            bw=100,
            delay="1ms",
            use_tbf=False
        )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    info("\nStarting network...\n")

    net.start()

    info("\n================================\n")
    info("   ABILENE NETWORK STARTED\n")
    info("================================\n")

    info(
        f"Switches: {len(net.switches)}\n"
    )

    info(
        f"Hosts: {len(net.hosts)}\n"
    )

    info(
        "\nRun:\n"
        "  nodes\n"
        "  net\n"
        "  h1 ping -c 3 h2\n\n"
    )

    CLI(net)

    net.stop()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    setLogLevel("info")

    create_network()
