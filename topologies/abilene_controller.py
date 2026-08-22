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

import networkx as nx


from ryu.lib import hub

from monitoring.network_state import NetworkStateTracker


class AbileneController(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # ==================================================
        # CONNECTED SWITCHES
        # ==================================================

        self.datapaths = {}

        # ==================================================
        # MAC LEARNING
        #
        # MAC -> (switch, port)
        # ==================================================

        self.mac_to_port = {}

        # ==================================================
        # HOST INFORMATION
        #
        # Abilene mapping:
        #
        # h1 -> New York -> s1 -> port 1
        # h2 -> Los Angeles -> s6 -> port 1
        # ==================================================

        self.hosts = {
            "10.0.0.1": {
                "switch": 1,
                "port": 1
            },

            "10.0.0.2": {
                "switch": 6,
                "port": 1
            }
        }

        # ==================================================
        # NETWORK MONITORING TRACKER
        # ==================================================

        self.tracker = NetworkStateTracker(default_capacity_mbps=100.0)
        self.monitor_thread = hub.spawn(self._monitor_loop)

        print("\n================================")
        print("     ABILENE RYU CONTROLLER")
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

        print(
            f"Switch connected: s{dpid}"
        )

        parser = datapath.ofproto_parser

        ofproto = datapath.ofproto

        # ------------------------------------------------
        # Table miss
        # ------------------------------------------------

        match = parser.OFPMatch()

        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER
            )
        ]

        self.add_flow(
            datapath,
            priority=0,
            match=match,
            actions=actions
        )

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

        req = parser.OFPPortStatsRequest(
            datapath,
            0,
            ofproto.OFPP_ANY
        )

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

        # Update link metrics from active topology links
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

    def add_flow(
        self,
        datapath,
        priority,
        match,
        actions
    ):

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

        switches = get_switch(
            self,
            None
        )

        links = get_link(
            self,
            None
        )

        graph = nx.DiGraph()

        # ------------------------------------------------
        # Add switches
        # ------------------------------------------------

        for switch in switches:

            graph.add_node(
                switch.dp.id
            )

        # ------------------------------------------------
        # Add directed links
        # ------------------------------------------------

        for link in links:

            source = link.src.dpid

            destination = link.dst.dpid

            source_port = link.src.port_no

            graph.add_edge(
                source,
                destination,
                port=source_port
            )

        return graph

    # ==================================================
    # FIND NEXT HOP
    # ==================================================

    def get_next_port(
        self,
        current_switch,
        destination_switch
    ):

        graph = self.get_network_graph()

        # ------------------------------------------------
        # Check that switches exist
        # ------------------------------------------------

        if current_switch not in graph:

            print(
                f"s{current_switch} "
                f"is not in topology"
            )

            return None

        if destination_switch not in graph:

            print(
                f"s{destination_switch} "
                f"is not in topology"
            )

            return None

        # ------------------------------------------------
        # Calculate shortest path
        # ------------------------------------------------

        try:

            path = nx.shortest_path(
                graph,
                current_switch,
                destination_switch
            )

        except nx.NetworkXNoPath:

            print(
                f"No path from "
                f"s{current_switch} "
                f"to s{destination_switch}"
            )

            return None

        print(
            f"Route selected: {path}"
        )

        # ------------------------------------------------
        # Already at destination
        # ------------------------------------------------

        if len(path) == 1:

            return None

        # ------------------------------------------------
        # Get next switch
        # ------------------------------------------------

        next_switch = path[1]

        # ------------------------------------------------
        # Get output port
        # ------------------------------------------------

        out_port = graph[
            current_switch
        ][
            next_switch
        ]["port"]

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

        # ------------------------------------------------
        # Parse packet
        # ------------------------------------------------

        pkt = packet.Packet(
            msg.data
        )

        eth = pkt.get_protocol(
            ethernet.ethernet
        )

        if eth is None:

            return

        # ==================================================
        # IGNORE LLDP
        # ==================================================

        if eth.ethertype == 0x88CC:

            return

        src_mac = eth.src

        dst_mac = eth.dst

        # ==================================================
        # LEARN SOURCE MAC (ACCESS PORTS ONLY)
        # ==================================================

        links = get_link(self, None)

        is_switch_port = any(
            link.src.dpid == dpid and link.src.port_no == in_port
            for link in links
        )

        if not is_switch_port:

            self.mac_to_port[src_mac] = (
                dpid,
                in_port
            )


        # ==================================================
        # CHECK FOR ARP
        # ==================================================

        arp_packet = pkt.get_protocol(
            arp.arp
        )

        if arp_packet is not None:

            target_ip = arp_packet.dst_ip

            # ------------------------------------------------
            # TARGET HOST IS KNOWN
            # ------------------------------------------------

            if target_ip in self.hosts:

                destination_switch = (
                    self.hosts[target_ip]["switch"]
                )

                destination_port = (
                    self.hosts[target_ip]["port"]
                )

                # ------------------------------------------------
                # ARP reached destination switch
                # ------------------------------------------------

                if dpid == destination_switch:

                    actions = [
                        parser.OFPActionOutput(
                            destination_port
                        )
                    ]

                    print(
                        f"ARP delivered to "
                        f"{target_ip} "
                        f"via s{dpid} "
                        f"port {destination_port}"
                    )

                # ------------------------------------------------
                # ARP still travelling through network
                # ------------------------------------------------

                else:

                    out_port = self.get_next_port(
                        dpid,
                        destination_switch
                    )

                    if out_port is None:

                        return

                    actions = [
                        parser.OFPActionOutput(
                            out_port
                        )
                    ]

            else:

                # ------------------------------------------------
                # Unknown ARP target.
                #
                # For this project we simply ignore it rather
                # than flooding a looped topology.
                # ------------------------------------------------

                return

        # ==================================================
        # NORMAL IP / ETHERNET PACKET
        # ==================================================

        else:

            # ------------------------------------------------
            # Destination MAC is already known
            # ------------------------------------------------

            if dst_mac in self.mac_to_port:

                destination_switch, destination_port = (
                    self.mac_to_port[dst_mac]
                )

                # Destination is on current switch

                if dpid == destination_switch:

                    actions = [
                        parser.OFPActionOutput(
                            destination_port
                        )
                    ]

                else:

                    out_port = self.get_next_port(
                        dpid,
                        destination_switch
                    )

                    if out_port is None:

                        return

                    actions = [
                        parser.OFPActionOutput(
                            out_port
                        )
                    ]

            else:

                # ------------------------------------------------
                # Unknown destination.
                #
                # Don't flood through Abilene.
                # ------------------------------------------------

                return

        # ==================================================
        # SEND PACKET
        # ==================================================

        if not actions:

            return

        # ------------------------------------------------
        # Install flow
        # ------------------------------------------------

        match = parser.OFPMatch(
            in_port=in_port,
            eth_dst=dst_mac
        )

        self.add_flow(
            datapath,
            priority=10,
            match=match,
            actions=actions
        )

        # ------------------------------------------------
        # Send current packet
        # ------------------------------------------------

        if msg.buffer_id == ofproto.OFP_NO_BUFFER:

            data = msg.data

        else:

            data = None

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )

        datapath.send_msg(out)


if __name__ == "__main__":

    app_manager.require_app(
        "ryu.controller.ofp_handler"
    )
