"""Isolated two-path Mininet demo with explicit 100 Mbps traffic shaping."""
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.log import setLogLevel, info


class LiveDemoTopo(Topo):
    def build(self):
        h1 = self.addHost("h1", ip="10.0.0.1/24")
        h2 = self.addHost("h2", ip="10.0.0.2/24")
        s1, s2, s3, s4 = (self.addSwitch(f"s{i}") for i in range(1, 5))
        self.addLink(h1, s1, bw=100)
        self.addLink(h2, s4, bw=100)
        for left, right in ((s1, s2), (s2, s4), (s1, s3), (s3, s4)):
            self.addLink(left, right, bw=100)


def run():
    net = Mininet(topo=LiveDemoTopo(), controller=None, switch=OVSSwitch,
                  link=TCLink, autoSetMacs=True, build=False)
    net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)
    net.build()
    net.start()
    try:
        for switch in net.switches:
            switch.cmd("ovs-vsctl set bridge %s protocols=OpenFlow13" % switch.name)
        info("\n*** Live SDN demo: h1 ping -c 5 h2; use links and ovs-ofctl dump-flows s1\n")
        CLI(net)
    finally:
        net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    run()
