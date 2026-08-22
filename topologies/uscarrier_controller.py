from ryu.base import app_manager

from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls
)

from ryu.ofproto import ofproto_v1_3

from ryu.topology.api import get_switch, get_link

from ryu.lib.packet import (
    packet,
    ethernet,
    arp
)

from ryu.lib import hub

import networkx as nx
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from monitoring.network_state import NetworkStateTracker


class UsCarrierController(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # ==================================================
        # CONNECTED SWITCHES
        # ==================================================

        self.datapaths = {}

        # ==================================================
        # MAC LEARNING: MAC -> (switch_dpid, port)
        # ==================================================

        self.mac_to_port = {}

        # ==================================================
        # HOST INFORMATION
        #
        # UsCarrier mapping:
        #   h1 -> Fort Myers  -> s41  (dpid=41)  -> port 1
        #   h2 -> Carlisle    -> s148 (dpid=148) -> port 1
        # ==================================================

        self.hosts = {
            "10.0.0.1": {
                "switch": 41,
                "port": 1
            },
            "10.0.0.2": {
                "switch": 148,
                "port": 1
            }
        }

        # ==================================================
        # NETWORK MONITORING TRACKER
        # ==================================================

        self.tracker = NetworkStateTracker(default_capacity_mbps=100.0)
        self.monitor_thread = hub.spawn(self._monitor_loop)

        print("\n================================")
        print("   USCARRIER RYU CONTROLLER")
        print("   158 nodes / 189 edges")
        print("   h1: Fort Myers  (s41)")
        print("   h2: Carlisle    (s148)")
        print("================================\n")

    # ==================================================
    # SWITCH CONNECTION
    # ==================================================

    @set_ev_cls(
        ofp_event.EventOFPSwitchFeatures,
        CONFIG_DISPATCHER
    )
    def switch_features_handler(self, ev):

        datapath = ev.msg.datapath
        dpid = datapath.id
        self.datapaths[dpid] = datapath

        print(f"Switch connected: s{dpid}")

        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        # Table-miss: send to controller
        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER
            )
        ]
        self.add_flow(datapath, priority=0, match=match, actions=actions)

    # ==================================================
    # PERIODIC MONITORING LOOP
    # ==================================================

    def _monitor_loop(self):

        while True:
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)
            hub.sleep(2)

    def _request_stats(self, datapath):

        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    # ==================================================
    # PORT STATS REPLY HANDLER
    # ==================================================

    @set_ev_cls(
        ofp_event.EventOFPPortStatsReply,
        MAIN_DISPATCHER
    )
    def port_stats_reply_handler(self, ev):

        body = ev.msg.body
        dpid = ev.msg.datapath.id

        for stat in body:
            self.tracker.update_port_stats(
                dpid=dpid,
                port_no=stat.port_no,
                tx_bytes=stat.tx_bytes,
                rx_bytes=stat.rx_bytes,
                tx_packets=stat.tx_packets,
                rx_packets=stat.rx_packets,
                tx_errors=stat.tx_errors,
                rx_dropped=stat.rx_dropped
            )

        links = get_link(self, None)
        self.tracker.update_link_topology(links)

    # ==================================================
    # GET STRUCTURED NETWORK STATE
    # ==================================================

    def get_network_state(self):

        return self.tracker.get_structured_state()

    # ==================================================
    # ADD FLOW
    # ==================================================

    def add_flow(self, datapath, priority, match, actions):

        parser = datapath.ofproto_parser
        instructions = [
            parser.OFPInstructionActions(
                datapath.ofproto.OFPIT_APPLY_ACTIONS,
                actions
            )
        ]
        flow_mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=instructions
        )
        datapath.send_msg(flow_mod)

    # ==================================================
    # GET CURRENT NETWORK GRAPH
    # ==================================================

    def get_network_graph(self):

        switches = get_switch(self, None)
        links = get_link(self, None)

        graph = nx.DiGraph()

        for switch in switches:
            graph.add_node(switch.dp.id)

        for link in links:
            graph.add_edge(
                link.src.dpid,
                link.dst.dpid,
                port=link.src.port_no
            )

        return graph

    # ==================================================
    # GET NEXT HOP PORT
    # ==================================================

    def get_next_port(self, current_switch, destination_switch):

        graph = self.get_network_graph()

        if current_switch not in graph:
            print(f"s{current_switch} not in topology")
            return None

        if destination_switch not in graph:
            print(f"s{destination_switch} not in topology")
            return None

        try:
            path = nx.shortest_path(graph, current_switch, destination_switch)
        except nx.NetworkXNoPath:
            print(f"No path: s{current_switch} -> s{destination_switch}")
            return None

        print(f"Route selected: {path}")

        if len(path) == 1:
            return None

        next_switch = path[1]
        out_port = graph[current_switch][next_switch]["port"]

        return out_port

    # ==================================================
    # PACKET IN
    # ==================================================

    @set_ev_cls(
        ofp_event.EventOFPPacketIn,
        MAIN_DISPATCHER
    )
    def packet_in_handler(self, ev):

        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth is None:
            return

        # Ignore LLDP
        if eth.ethertype == 0x88CC:
            return

        src_mac = eth.src
        dst_mac = eth.dst

        # Learn MAC only on access (non-switch) ports
        links = get_link(self, None)
        is_switch_port = any(
            link.src.dpid == dpid and link.src.port_no == in_port
            for link in links
        )
        if not is_switch_port:
            self.mac_to_port[src_mac] = (dpid, in_port)

        actions = None

        # ARP handling
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt is not None:
            target_ip = arp_pkt.dst_ip

            if target_ip in self.hosts:
                dst_switch = self.hosts[target_ip]["switch"]
                dst_port = self.hosts[target_ip]["port"]

                if dpid == dst_switch:
                    actions = [parser.OFPActionOutput(dst_port)]
                    print(f"ARP delivered to {target_ip} via s{dpid} port {dst_port}")
                else:
                    out_port = self.get_next_port(dpid, dst_switch)
                    if out_port is None:
                        return
                    actions = [parser.OFPActionOutput(out_port)]
            else:
                return

        # IP/Ethernet forwarding
        else:
            if dst_mac in self.mac_to_port:
                dst_switch, dst_port = self.mac_to_port[dst_mac]

                if dpid == dst_switch:
                    actions = [parser.OFPActionOutput(dst_port)]
                else:
                    out_port = self.get_next_port(dpid, dst_switch)
                    if out_port is None:
                        return
                    actions = [parser.OFPActionOutput(out_port)]
            else:
                return

        if not actions:
            return

        # Install flow
        match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac)
        self.add_flow(datapath, priority=10, match=match, actions=actions)

        # Send packet
        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)


if __name__ == "__main__":
    app_manager.require_app("ryu.controller.ofp_handler")
