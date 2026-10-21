
#!/usr/bin/env python3

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.log import setLogLevel, info


class MultiPathTopo(Topo):
    def build(self):
        h1 = self.addHost("h1", ip="10.0.0.1/24")
        h2 = self.addHost("h2", ip="10.0.0.2/24")

        s1 = self.addSwitch("s1")
        s2 = self.addSwitch("s2")
        s3 = self.addSwitch("s3")
        s4 = self.addSwitch("s4")

        # Host access links
        self.addLink(h1, s1)
        self.addLink(h2, s4)

        # Two alternative paths: s1-s2-s4 and s1-s3-s4
        self.addLink(s1, s2)
        self.addLink(s2, s4)
        self.addLink(s1, s3)
        self.addLink(s3, s4)


def run():
    topo = MultiPathTopo()

    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        autoSetMacs=True,
        build=False
    )

    net.addController(
        "c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6653
    )

    net.build()
    net.start()

    try:
        for switch in net.switches:
            switch.cmd("ovs-vsctl set bridge %s protocols=OpenFlow13" %
                       switch.name)

        info("\n*** Network started. Testing connectivity...\n")
        net.pingAll()

        info("\n*** You can now run commands in the Mininet CLI.\n")
        info("*** Example: h1 ping -c 5 h2\n")
        info("*** Example: links\n")
        info("*** Example: exit\n\n")

        CLI(net)

    finally:
        net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    run()
