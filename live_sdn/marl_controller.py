"""Isolated OpenFlow 1.3 controller for the repository Mininet demo.

Runtime policy is a deterministic shortest-path fallback, optionally weighted
by measured utilization once a capacity and counter deltas are available.
Current DQN checkpoints are deliberately not invoked; see README.md.
"""
from __future__ import annotations

import json
import os
import time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls,
)
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, arp
from ryu.topology import event
from ryu.topology.api import get_switch, get_link

from live_sdn.routing_adapter import build_live_graph, choose_path, path_outputs


class LiveRoutingController(app_manager.RyuApp):
    """Reactive host-to-host forwarding with topology and port-stat tracking."""
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    STATS_PERIOD = 2.0
    STATS_MAX_AGE = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.mac_locations = {}  # mac -> (dpid, host-facing port)
        self.ip_locations = {}   # ip -> (dpid, host-facing port)
        self.failed_links = set()
        self.port_samples = {}   # (dpid, port) -> (bytes, monotonic time)
        self.port_metrics = {}   # (dpid, port) -> measured utilization metadata
        self.capacity_mbps = self._read_capacity()
        self.congestion_aware = os.environ.get("LIVE_CONGESTION_ROUTING", "0") == "1"
        self.monitor = hub.spawn(self._monitor)
        self.logger.info(
            "LiveRoutingController ready: OpenFlow 1.3; policy=%s; capacity=%s Mbps; DQN=disabled (%s)",
            "measured-utilization shortest path" if self.congestion_aware else "shortest path",
            self.capacity_mbps,
            "simulator observation/action contract is not available on live switches",
        )

    @staticmethod
    def _read_capacity():
        raw = os.environ.get("LIVE_LINK_CAPACITY_MBPS", "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError("LIVE_LINK_CAPACITY_MBPS must be numeric") from exc
        if value <= 0:
            raise ValueError("LIVE_LINK_CAPACITY_MBPS must be positive")
        return value

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features(self, ev):
        dp = ev.msg.datapath
        self.datapaths[dp.id] = dp
        parser, ofp = dp.ofproto_parser, dp.ofproto
        self._add_flow(
            dp, 0, parser.OFPMatch(),
            [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)],
            cookie=0
        )
        self.logger.info("OpenFlow 1.3 switch connected dpid=%s", dp.id)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def datapath_state(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)

    @set_ev_cls(event.EventLinkDelete)
    def link_deleted(self, ev):
        link = ev.link
        self.failed_links.add(tuple(sorted((int(link.src.dpid), int(link.dst.dpid)))))
        self._flush_route_flows()
        self.logger.warning("Link removed from discovered topology: %s<->%s", link.src.dpid, link.dst.dpid)

    @set_ev_cls(event.EventLinkAdd)
    def link_added(self, ev):
        link = ev.link
        key = tuple(sorted((int(link.src.dpid), int(link.dst.dpid))))
        self.failed_links.discard(key)
        self.logger.info("Link discovered: %s<->%s", link.src.dpid, link.dst.dpid)

    def _flush_route_flows(self):
        """Invalidate only this controller's forwarding entries after topology change."""
        for dp in list(self.datapaths.values()):
            ofp, parser = dp.ofproto, dp.ofproto_parser
            dp.send_msg(parser.OFPFlowMod(
                datapath=dp, command=ofp.OFPFC_DELETE,
                cookie=0x53444E4D41524C, cookie_mask=0xFFFFFFFFFFFFFFFF,
                table_id=ofp.OFPTT_ALL, out_port=ofp.OFPP_ANY,
                out_group=ofp.OFPG_ANY, priority=20,
                match=parser.OFPMatch()))
        self.logger.info("Invalidated controller forwarding flows after topology change")

    @set_ev_cls(event.EventSwitchLeave)
    def switch_left(self, ev):
        self.datapaths.pop(ev.switch.dp.id, None)

    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                try:
                    dp.send_msg(dp.ofproto_parser.OFPPortStatsRequest(dp, 0, dp.ofproto.OFPP_ANY))
                except Exception as exc:  # keep controller alive during disconnects
                    self.logger.debug("Port stats request failed dpid=%s: %s", dp.id, exc)
            hub.sleep(self.STATS_PERIOD)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply(self, ev):
        if self.capacity_mbps is None:
            return
        now = time.monotonic()
        dpid = ev.msg.datapath.id
        for stat in ev.msg.body:
            key = (dpid, int(stat.port_no))
            previous = self.port_samples.get(key)
            self.port_samples[key] = (int(stat.tx_bytes), now)
            if previous is None:
                continue
            byte_delta = int(stat.tx_bytes) - previous[0]
            elapsed = now - previous[1]
            if byte_delta < 0 or elapsed <= 0:
                continue
            mbps = byte_delta * 8.0 / elapsed / 1_000_000.0
            self.port_metrics[key] = {
                "utilization": min(1.0, mbps / self.capacity_mbps),
                "capacity_mbps": self.capacity_mbps,
                "throughput_mbps": mbps,
                "sampled_at": now,
                "source": "OpenFlow tx_bytes counter delta / configured Mininet capacity",
            }

    def _graph(self):
        switches = get_switch(self, None)
        links = get_link(self, None)
        metrics = {}
        now = time.monotonic()
        for link in links:
            key = (int(link.src.dpid), int(link.src.port_no))
            m = self.port_metrics.get(key)
            if m and now - m["sampled_at"] <= self.STATS_MAX_AGE:
                metrics[(int(link.src.dpid), int(link.dst.dpid))] = m
        return build_live_graph(switches, links, self.failed_links, metrics)

    def _route(self, src_dpid, dst_dpid):
        graph = self._graph()
        path = choose_path(graph, src_dpid, dst_dpid, congestion_aware=self.congestion_aware)
        if path is None:
            self.logger.warning("No live path from s%s to s%s", src_dpid, dst_dpid)
            return None
        hops = path_outputs(graph, path)
        metric_note = "measured utilization" if self.congestion_aware and all(
            "utilization" in graph[u][v] for u, v in zip(path, path[1:])
        ) else "topology-only shortest path"
        self.logger.info("Selected route %s via %s (%s)", path, metric_note, "rule-based fallback")
        return graph, path, hops

    def _install_path(self, graph, path, src_mac, dst_mac, src_location, dst_location):
        """Install forward and reverse path rules using discovered port mapping."""
        forward = path_outputs(graph, path)
        reverse = path_outputs(graph, list(reversed(path)))
        entries = []
        if forward:
            entries.extend((dpid, out_port, src_mac, dst_mac) for dpid, out_port in forward)
            entries.append((path[-1], dst_location[1], src_mac, dst_mac))
        else:
            entries.append((path[0], dst_location[1], src_mac, dst_mac))
        if reverse:
            entries.extend((dpid, out_port, dst_mac, src_mac) for dpid, out_port in reverse)
            entries.append((path[0], src_location[1], dst_mac, src_mac))
        else:
            entries.append((path[0], src_location[1], dst_mac, src_mac))
        for dpid, out_port, eth_src, eth_dst in entries:
            dp = self.datapaths.get(int(dpid))
            if dp is None:
                continue
            parser = dp.ofproto_parser
            match = parser.OFPMatch(eth_src=eth_src, eth_dst=eth_dst)
            self._add_flow(dp, 20, match, [parser.OFPActionOutput(int(out_port))], idle_timeout=60)
        self.logger.info("Issued %d flow mods for bidirectional route %s", len(entries), path)

    def _add_flow(self, dp, priority, match, actions, idle_timeout=0,
              cookie=0x53444E4D41524C):
        parser, ofp = dp.ofproto_parser, dp.ofproto
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            cookie=cookie
        ))

    def _flood_arp_to_hosts(self, msg, ingress_dpid, ingress_port):
        """Flood ARP only to learned/possible host-facing ports, never over links."""
        graph = self._graph()
        inter_switch = {(dpid, data["out_port"]) for dpid, _, data in graph.edges(data=True)}
        for dpid, dp in self.datapaths.items():
            for port in dp.ports:
                port_no = int(port)
                if port_no > dp.ofproto.OFPP_MAX or (dpid, port_no) in inter_switch:
                    continue
                if dpid == ingress_dpid and port_no == ingress_port:
                    continue
                dp.send_msg(dp.ofproto_parser.OFPPacketOut(
                    datapath=dp, buffer_id=dp.ofproto.OFP_NO_BUFFER,
                    in_port=dp.ofproto.OFPP_CONTROLLER,
                    actions=[dp.ofproto_parser.OFPActionOutput(port_no)], data=msg.data))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg, dp = ev.msg, ev.msg.datapath
        dpid, in_port = int(dp.id), int(ev.msg.match["in_port"])
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None or eth.ethertype == 0x88CC:
            return
        graph = self._graph()
        inter_switch = {(n, data["out_port"]) for n, _, data in graph.edges(data=True)}
        if (dpid, in_port) not in inter_switch:
            self.mac_locations[eth.src] = (dpid, in_port)
            arp_pkt = pkt.get_protocol(arp.arp)
            if arp_pkt:
                self.ip_locations[arp_pkt.src_ip] = (dpid, in_port)
        arp_pkt = pkt.get_protocol(arp.arp)
        dst_location = self.mac_locations.get(eth.dst)
        if arp_pkt and arp_pkt.dst_ip in self.ip_locations:
            dst_location = self.ip_locations[arp_pkt.dst_ip]
        if dst_location is None:
            if arp_pkt:
                self._flood_arp_to_hosts(msg, dpid, in_port)
            return
        src_location = self.mac_locations.get(eth.src)
        if src_location is None:
            return
        route = self._route(src_location[0], dst_location[0])
        if route is None:
            return
        route_graph, path, _ = route
        self._install_path(route_graph, path, eth.src, eth.dst, src_location, dst_location)
        # Initial Packet-Out at ingress; subsequent packets use installed flows.
        position = path.index(dpid) if dpid in path else None
        if position is not None and position < len(path) - 1:
            out_port = int(route_graph[path[position]][path[position + 1]]["out_port"])
        elif dpid == dst_location[0]:
            out_port = dst_location[1]
        else:
            return
        dp.send_msg(dp.ofproto_parser.OFPPacketOut(
            datapath=dp, buffer_id=msg.buffer_id, in_port=in_port,
            actions=[dp.ofproto_parser.OFPActionOutput(out_port)],
            data=msg.data if msg.buffer_id == dp.ofproto.OFP_NO_BUFFER else None))
