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
    arp,
    ipv4
)

from ryu.lib import hub

import networkx as nx
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from monitoring.network_state import NetworkStateTracker


class GenericSDNController(app_manager.RyuApp):
    """
    Universal SDN Controller capable of running on ANY topology from
    Internet Topology Zoo (or any custom GML network) with ZERO code changes.

    Features:
    1. Hybrid Topology Discovery:
       - Uses GML topology graph (if TOPOLOGY_GML env var set) for instantaneous 100% complete path computation.
       - Dynamic OpenFlow LLDP discovery for real-time link monitoring and state tracking.
    2. Dynamic Host Discovery (learns host IP & MAC on access ports automatically).
    3. Graph-based Shortest Path Routing using NetworkX.
    4. Real-time Network Monitoring & State Tracking via NetworkStateTracker.
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.datapaths = {}

        # Dynamic Host Learning:
        # mac_to_location: mac -> (dpid, port)
        # ip_to_location: ip -> (dpid, port)
        self.mac_to_location = {}
        self.ip_to_location = {}

        # Custom Link Penalty Weights set by RL Congestion Agent
        self.custom_link_weights = {}

        # Static / Hybrid Graph Initialization if TOPOLOGY_GML environment variable is provided
        self.static_graph = None
        self.nodes_list = []
        gml_env = os.environ.get("TOPOLOGY_GML", "")
        if gml_env:
            self._load_static_gml(gml_env)

        # Real-Time Monitoring Tracker
        self.tracker = NetworkStateTracker(default_capacity_mbps=100.0)
        self.monitor_thread = hub.spawn(self._monitor_loop)

        print("\n==================================================")
        print("    GENERIC SDN MARL-READY RYU CONTROLLER         ")
        if self.static_graph:
            print(f"    Loaded Topology: {gml_env} ({self.static_graph.number_of_nodes()} nodes, {self.static_graph.number_of_edges()} edges)")
        else:
            print("    Mode: Pure Dynamic LLDP Topology Discovery   ")
        print("==================================================\n")

    def update_link_weights(self, weights):
        """
        Allows external RL Agent (Congestion Agent) to dynamically set
        custom routing penalty weights on specific links.
        """
        new_weights = weights if weights is not None else {}
        if new_weights != self.custom_link_weights:
            self.custom_link_weights = new_weights
            self.flush_flow_rules()

    def flush_flow_rules(self):
        """
        Flushes active forwarding rules (priority=10) on all datapaths so switches
        immediately request new paths using updated RL link weights.
        """
        for dp in list(self.datapaths.values()):
            try:
                parser = dp.ofproto_parser
                ofproto = dp.ofproto
                mod = parser.OFPFlowMod(
                    datapath=dp,
                    command=ofproto.OFPFC_DELETE,
                    out_port=ofproto.OFPP_ANY,
                    out_group=ofproto.OFPG_ANY,
                    priority=10
                )
                dp.send_msg(mod)
            except Exception:
                pass


    def _load_static_gml(self, gml_name):
        zoo_dir = os.path.join(os.path.dirname(__file__), "InternetTopologyZoo/gml")
        if not gml_name.endswith(".gml"):
            gml_name += ".gml"

        filepath = os.path.join(zoo_dir, gml_name) if not os.path.exists(gml_name) else gml_name

        if os.path.exists(filepath):
            try:
                g = nx.read_gml(filepath, label="id")
                ug = g.to_undirected()
                if not nx.is_connected(ug):
                    comps = list(nx.connected_components(ug))
                    largest = max(comps, key=len)
                    g = g.subgraph(largest).copy()
                    ug = g.to_undirected()

                self.nodes_list = list(g.nodes())
                # Build switch-indexed graph (dpid = index + 1)
                self.static_graph = nx.DiGraph()
                for u, v in ug.edges():
                    dpid_u = self.nodes_list.index(u) + 1
                    dpid_v = self.nodes_list.index(v) + 1
                    self.static_graph.add_edge(dpid_u, dpid_v)
                    self.static_graph.add_edge(dpid_v, dpid_u)
                print(f"[Topology Loaded] Successfully initialized graph for {gml_name}")
            except Exception as e:
                print(f"[Topology Load Error] {e}")

    # ==================================================
    # SWITCH CONNECTION HANDLER
    # ==================================================

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        dpid = datapath.id
        self.datapaths[dpid] = datapath

        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        # Install table-miss flow rule: send unknown packets to controller
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
        import json
        while True:
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)

            # File IPC: Expose live state to external agents
            try:
                state_data = self.get_network_state()
                with open("/tmp/sdn_network_state.json", "w") as f:
                    json.dump(state_data, f)
            except Exception as e:
                pass

            # File IPC: Read external link weight actions from agents
            try:
                if os.path.exists("/tmp/sdn_link_weights.json"):
                    with open("/tmp/sdn_link_weights.json", "r") as f:
                        weights = json.load(f)
                        self.custom_link_weights = weights
            except Exception as e:
                pass

            hub.sleep(2)

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
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

    def get_network_state(self):
        return self.tracker.get_structured_state()

    # ==================================================
    # OPENFLOW UTILITIES
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
    # DYNAMIC GRAPH DISCOVERY & ROUTING
    # ==================================================

    def get_network_graph(self):
        switches = get_switch(self, None)
        links = get_link(self, None)

        graph = nx.DiGraph()

        for switch in switches:
            graph.add_node(switch.dp.id)

        for link in links:
            link_key = f"s{link.src.dpid}_p{link.src.port_no}->s{link.dst.dpid}_p{link.dst.port_no}"
            alt_key = f"s{link.src.dpid}->s{link.dst.dpid}"
            weight = float(self.custom_link_weights.get(link_key, self.custom_link_weights.get(alt_key, 1.0)))

            graph.add_edge(
                link.src.dpid,
                link.dst.dpid,
                port=link.src.port_no,
                weight=weight
            )

        return graph

    def get_next_port(self, current_switch, destination_switch):
        # 1. Try dynamic LLDP graph first
        graph = self.get_network_graph()

        if current_switch in graph and destination_switch in graph:
            try:
                path = nx.shortest_path(graph, current_switch, destination_switch, weight="weight")
                if len(path) > 1:
                    next_switch = path[1]
                    return graph[current_switch][next_switch]["port"]
            except nx.NetworkXNoPath:
                pass

        # 2. Fallback to static GML graph port mapping if available
        if self.static_graph and current_switch in self.static_graph and destination_switch in self.static_graph:
            try:
                path = nx.shortest_path(self.static_graph, current_switch, destination_switch)
                if len(path) > 1:
                    next_switch = path[1]
                    # Find out port connecting current_switch to next_switch via LLDP links or convention
                    links = get_link(self, None)
                    for link in links:
                        if link.src.dpid == current_switch and link.dst.dpid == next_switch:
                            return link.src.port_no
            except nx.NetworkXNoPath:
                pass

        return None

    def is_inter_switch_port(self, dpid, port_no):
        links = get_link(self, None)
        return any(
            link.src.dpid == dpid and link.src.port_no == port_no
            for link in links
        )

    # ==================================================
    # PACKET IN HANDLING (DYNAMIC HOST DISCOVERY & ROUTING)
    # ==================================================

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth is None or eth.ethertype == 0x88CC:  # Drop LLDP
            return

        src_mac = eth.src
        dst_mac = eth.dst

        # Dynamic Host Learning on Access (Non-Switch) Ports
        if not self.is_inter_switch_port(dpid, in_port):
            self.mac_to_location[src_mac] = (dpid, in_port)

            # Learn IP if packet is ARP or IPv4
            arp_pkt = pkt.get_protocol(arp.arp)
            if arp_pkt:
                self.ip_to_location[arp_pkt.src_ip] = (dpid, in_port)

            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            if ip_pkt:
                self.ip_to_location[ip_pkt.src] = (dpid, in_port)

        actions = None

        # --- ARP PACKET HANDLING ---
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt is not None:
            target_ip = arp_pkt.dst_ip

            if target_ip in self.ip_to_location:
                dst_switch, dst_port = self.ip_to_location[target_ip]

                if dpid == dst_switch:
                    actions = [parser.OFPActionOutput(dst_port)]
                else:
                    out_port = self.get_next_port(dpid, dst_switch)
                    if out_port is None:
                        return
                    actions = [parser.OFPActionOutput(out_port)]
            else:
                # Flood ARP ONLY on access ports across all connected switches
                for dp in self.datapaths.values():
                    dp_parser = dp.ofproto_parser
                    dp_ofproto = dp.ofproto
                    for port_no in dp.ports:
                        if port_no <= dp_ofproto.OFPP_MAX and not self.is_inter_switch_port(dp.id, port_no):
                            if dp.id == dpid and port_no == in_port:
                                continue
                            out = dp_parser.OFPPacketOut(
                                datapath=dp,
                                buffer_id=dp_ofproto.OFP_NO_BUFFER,
                                in_port=dp_ofproto.OFPP_CONTROLLER,
                                actions=[dp_parser.OFPActionOutput(port_no)],
                                data=msg.data
                            )
                            dp.send_msg(out)
                return

        # --- IP & ETHERNET FORWARDING ---
        else:
            if dst_mac in self.mac_to_location:
                dst_switch, dst_port = self.mac_to_location[dst_mac]

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

        # Install reactive flow rule
        match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac)
        self.add_flow(datapath, priority=10, match=match, actions=actions)

        # Forward packet
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
