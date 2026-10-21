"""Configurable eight-switch, three-route Mininet/OVS live test network."""
import argparse

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import Topo


class ParallelRouteTopo(Topo):
    def __init__(self, capacity_mbps=100.0, **opts):
        self.capacity_mbps = float(capacity_mbps)
        if self.capacity_mbps <= 0:
            raise ValueError("capacity_mbps must be positive")
        super().__init__(**opts)

    def build(self):
        h1 = self.addHost("h1", ip="10.0.0.1/24")
        h2 = self.addHost("h2", ip="10.0.0.2/24")
        switches = {i: self.addSwitch(f"s{i}", protocols="OpenFlow13") for i in range(1, 9)}
        bw = self.capacity_mbps
        self.addLink(h1, switches[1], bw=bw)
        self.addLink(h2, switches[8], bw=bw)
        routes = ((1, 2, 5, 8), (1, 3, 6, 8), (1, 4, 7, 8))
        edges = {tuple(sorted(e)) for route in routes for e in zip(route, route[1:])}
        # Cross links provide extra next-hop candidates and rerouting choices.
        edges.update({(2, 6), (3, 7)})
        for u, v in sorted(edges):
            self.addLink(switches[u], switches[v], bw=bw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-mbps", type=float, default=100.0)
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--controller-port", type=int, default=6653)
    args = parser.parse_args()
    topo = ParallelRouteTopo(capacity_mbps=args.capacity_mbps)
    net = Mininet(topo=topo, controller=None, switch=OVSSwitch, link=TCLink,
                  autoSetMacs=True, build=False)
    net.addController("c0", controller=RemoteController,
                      ip=args.controller_ip, port=args.controller_port)
    net.build()
    net.start()
    try:
        for switch in net.switches:
            switch.cmd("ovs-vsctl set bridge %s protocols=OpenFlow13" % switch.name)
        info("\n*** marl_live: 100 Mbps configured per-link shaping; run 'h1 ping -c 5 h2'\n")
        CLI(net)
    finally:
        net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    main()
