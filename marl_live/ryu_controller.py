"""Ryu OpenFlow 1.3 app that applies trained candidate-routing policies."""
from __future__ import annotations

import os
import time

import networkx as nx

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import arp, ethernet, packet
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event
from ryu.topology.api import get_link, get_switch

from marl_live.orchestrator import RoutingOrchestrator
from marl_live.policy import RoutingPolicy
from marl_live.routing import build_policy_route


class MARLLiveController(app_manager.RyuApp):
    """Per-neighbor learned decisions, with explicit safety fallback labels."""
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    COOKIE = 0x4D41524C495645
    FLOW_TIMEOUT = 12
    ROUTE_CHANGE_COOLDOWN = 15.0
    MIN_UTIL_IMPROVEMENT = 0.10
    MAX_HOPS = 32
    STATS_PERIOD = 2.0
    MAX_COUNTER_AGE = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.hosts = {}  # MAC -> (dpid, access_port)
        self.host_ips = {}  # IPv4 -> (dpid, access_port)
        self.known_links = {}  # (src,dst) -> ports/status/metrics
        self.port_counters = {}
        self.route_cache = {}  # (src_mac,dst_mac) -> {path, changed_at}
        self.capacity_mbps = self._capacity()
        self.orchestrator = RoutingOrchestrator()
        self.congestion_policy = self._load_policy("MARL_LIVE_CONGESTION_POLICY")
        self.failure_policy = self._load_policy("MARL_LIVE_FAILURE_POLICY")
        self.graph = nx.DiGraph()
        self.monitor = hub.spawn(self._monitor)
        self.logger.info(
            "marl_live controller ready: OpenFlow13; congestion_policy=%s; failure_policy=%s; "
            "configured_capacity_mbps=%s",
            "loaded" if self.congestion_policy else "missing (fallback only)",
            "loaded" if self.failure_policy else "missing (fallback only)",
            self.capacity_mbps,
        )

    def _load_policy(self, env_name):
        path = os.environ.get(env_name)
        if not path:
            return None
        try:
            policy = RoutingPolicy.load(path, device="cpu")
            self.logger.info("Loaded %s schema=%s from %s", env_name,
                             policy.metadata.get("observation_schema"), path)
            return policy
        except Exception as exc:
            self.logger.error("Could not load %s at %s: %s", env_name, path, exc)
            return None

    @staticmethod
    def _capacity():
        raw = os.environ.get("MARL_LIVE_LINK_CAPACITY_MBPS")
        if raw is None:
            return None
        try:
            capacity = float(raw)
        except ValueError as exc:
            raise ValueError("MARL_LIVE_LINK_CAPACITY_MBPS must be numeric") from exc
        if capacity <= 0:
            raise ValueError("MARL_LIVE_LINK_CAPACITY_MBPS must be positive")
        return capacity

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features(self, ev):
        dp = ev.msg.datapath
        self.datapaths[dp.id] = dp
        self._add_flow(dp, 0, dp.ofproto_parser.OFPMatch(),
                       [dp.ofproto_parser.OFPActionOutput(dp.ofproto.OFPP_CONTROLLER,
                                                          dp.ofproto.OFPCML_NO_BUFFER)],
                       timeout=0)
        self.logger.info("OpenFlow13 handshake complete dpid=%s", dp.id)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def datapath_state(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter(self, _ev):
        self._sync_topology()

    @set_ev_cls(event.EventSwitchLeave)
    def switch_leave(self, ev):
        dpid = ev.switch.dp.id
        self.datapaths.pop(dpid, None)
        changed = False
        for edge, data in self.known_links.items():
            if dpid in edge and not data["is_failed"]:
                data["is_failed"] = True
                changed = True
        if changed:
            self._rebuild_graph()
            self._invalidate_routes()

    @set_ev_cls(event.EventLinkAdd)
    def link_add(self, _ev):
        self._sync_topology()

    @set_ev_cls(event.EventLinkDelete)
    def link_delete(self, ev):
        src, dst = int(ev.link.src.dpid), int(ev.link.dst.dpid)
        changed = False
        for edge in ((src, dst), (dst, src)):
            if edge in self.known_links and not self.known_links[edge]["is_failed"]:
                self.known_links[edge]["is_failed"] = True
                changed = True
        if changed:
            self._rebuild_graph()
            self._invalidate_routes()
            self.logger.warning("Observed failed/disappeared inter-switch link %s<->%s", src, dst)

    def _sync_topology(self):
        for switch in get_switch(self, None):
            self.graph.add_node(int(switch.dp.id))
        active = set()
        for link in get_link(self, None):
            src, dst = int(link.src.dpid), int(link.dst.dpid)
            edge = (src, dst)
            active.add(edge)
            data = self.known_links.setdefault(edge, {})
            data.update(out_port=int(link.src.port_no), peer_port=int(link.dst.port_no), is_failed=False)
            if self.capacity_mbps is not None:
                data["capacity_mbps"] = self.capacity_mbps
            self._refresh_edge_metrics(edge)
        # A link reappearing clears its failure marker. LinkDelete marks it
        # immediately; this sweep also detects links absent from discovery.
        changed = False
        for edge, data in self.known_links.items():
            if edge not in active and not data["is_failed"]:
                data["is_failed"] = True
                changed = True
        self._rebuild_graph()
        if changed:
            self._invalidate_routes()

    def _rebuild_graph(self):
        nodes = list(self.graph.nodes)
        rebuilt = nx.DiGraph()
        rebuilt.add_nodes_from(nodes)
        for (src, dst), data in self.known_links.items():
            rebuilt.add_edge(src, dst, **data)
        self.graph = rebuilt

    def _refresh_edge_metrics(self, edge):
        src, dst = edge
        data = self.known_links[edge]
        if self.capacity_mbps is not None:
            data["capacity_mbps"] = self.capacity_mbps
        sample = self.port_counters.get((src, data["out_port"]))
        if (
            sample
            and "utilization" in sample
            and self.capacity_mbps
            and time.monotonic() - sample["time"] <= self.MAX_COUNTER_AGE
        ):
            data["utilization"] = sample["utilization"]
        else:
            data["utilization"] = None

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats(self, ev):
        if self.capacity_mbps is None:
            return
        now = time.monotonic()
        dpid = int(ev.msg.datapath.id)
        for stat in ev.msg.body:
            key = (dpid, int(stat.port_no))
            old = self.port_counters.get(key)
            self.port_counters[key] = {"bytes": int(stat.tx_bytes), "time": now}
            if old is None:
                continue
            delta, elapsed = int(stat.tx_bytes) - old["bytes"], now - old["time"]
            if delta < 0 or elapsed <= 0:
                continue
            mbps = delta * 8.0 / elapsed / 1_000_000.0
            self.port_counters[key].update(utilization=mbps / self.capacity_mbps,
                                           throughput_mbps=mbps)
        for edge in self.known_links:
            self._refresh_edge_metrics(edge)
        self._rebuild_graph()

    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                try:
                    dp.send_msg(dp.ofproto_parser.OFPPortStatsRequest(dp, 0, dp.ofproto.OFPP_ANY))
                except Exception as exc:
                    self.logger.debug("stats request failed dpid=%s: %s", dp.id, exc)
            hub.sleep(self.STATS_PERIOD)

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def port_status(self, ev):
        msg, dp = ev.msg, ev.msg.datapath
        port_no = int(msg.desc.port_no)
        down = bool(msg.desc.state & dp.ofproto.OFPPS_LINK_DOWN)
        changed = False
        for (src, _dst), data in self.known_links.items():
            if src == int(dp.id) and data["out_port"] == port_no:
                if data["is_failed"] != down:
                    data["is_failed"] = down
                    changed = True
        if changed:
            self._rebuild_graph()
            self._invalidate_routes()
            self.logger.warning("Port state changed dpid=%s port=%s down=%s", dp.id, port_no, down)

    def _invalidate_routes(self):
        for dp in list(self.datapaths.values()):
            ofp, parser = dp.ofproto, dp.ofproto_parser
            dp.send_msg(parser.OFPFlowMod(
                datapath=dp, command=ofp.OFPFC_DELETE,
                cookie=self.COOKIE, cookie_mask=0xFFFFFFFFFFFFFFFF,
                table_id=ofp.OFPTT_ALL, out_port=ofp.OFPP_ANY,
                out_group=ofp.OFPG_ANY, match=parser.OFPMatch()))

            # Restore the table-miss rule so packets reach Ryu after invalidation.
            self._add_flow(
                dp, 0, parser.OFPMatch(),
                [parser.OFPActionOutput(
                    ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER
                )],
                timeout=0,
            )

    def _add_flow(self, dp, priority, match, actions, timeout=FLOW_TIMEOUT):
        parser, ofp = dp.ofproto_parser, dp.ofproto
        instructions = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(parser.OFPFlowMod(datapath=dp, cookie=self.COOKIE,
                                      priority=priority, match=match,
                                      instructions=instructions,
                                      idle_timeout=timeout, hard_timeout=timeout))

    def _build_path(self, source, destination):
        def log_decision(decision, obs):
            self.logger.info("policy_decision source=%s node=%s next=%s reason=%s features=%s",
                             decision.decision_source, obs["current_node"], decision.next_node,
                             decision.reason, obs["candidate_features"][decision.action_index].tolist())
        result = build_policy_route(self.graph, source, destination, self.orchestrator,
                                    self.congestion_policy, self.failure_policy,
                                    self.MAX_HOPS, on_decision=log_decision)
        if result is None:
            self.logger.warning("No safe policy path from %s to %s", source, destination)
        return result

    def _install_direction(self, path, src_mac, dst_mac, dst_location):
        for index, dpid in enumerate(path):
            dp = self.datapaths.get(int(dpid))
            if dp is None:
                return False
            if index < len(path) - 1:
                edge = self.graph[dpid][path[index + 1]]
                if edge.get("is_failed"):
                    return False
                out_port = edge["out_port"]
            else:
                out_port = dst_location[1]
            parser = dp.ofproto_parser
            match = parser.OFPMatch(eth_src=src_mac, eth_dst=dst_mac)
            self._add_flow(dp, 20, match, [parser.OFPActionOutput(int(out_port))])
        return True

    def _stabilize_route(self, flow_key, proposed, sources):
        """Apply failure bypass and a short cooldown/hysteresis to congestion updates."""
        previous = self.route_cache.get(flow_key)
        if previous is None:
            self.route_cache[flow_key] = {"path": list(proposed), "changed_at": time.monotonic()}
            return proposed, sources
        old = previous["path"]
        old_healthy = all(self.graph.has_edge(u, v)
                          and not self.graph[u][v].get("is_failed", False)
                          for u, v in zip(old, old[1:]))
        if not old_healthy:
            self.route_cache[flow_key] = {"path": list(proposed), "changed_at": time.monotonic()}
            self.logger.info("route_change reason=failed_old_path flow=%s old=%s new=%s",
                             flow_key, old, proposed)
            return proposed, sources
        if old == proposed:
            return proposed, sources
        now = time.monotonic()
        if now - previous["changed_at"] < self.ROUTE_CHANGE_COOLDOWN:
            self.logger.info("route_hold source=route_hysteresis flow=%s path=%s", flow_key, old)
            return old, sources + ["route_hysteresis_hold"]
        def util_cost(path):
            values = [self.graph[u][v].get("utilization") for u, v in zip(path, path[1:])]
            if any(value is None for value in values):
                return None
            return sum(float(value) for value in values)
        old_cost, new_cost = util_cost(old), util_cost(proposed)
        if old_cost is not None and new_cost is not None and old_cost - new_cost < self.MIN_UTIL_IMPROVEMENT:
            self.logger.info("route_hold source=route_hysteresis flow=%s path=%s improvement=%s",
                             flow_key, old, old_cost - new_cost)
            return old, sources + ["route_hysteresis_hold"]
        self.route_cache[flow_key] = {"path": list(proposed), "changed_at": now}
        self.logger.info("route_change reason=policy_update flow=%s old=%s new=%s",
                         flow_key, old, proposed)
        return proposed, sources

    def _flood_arp_access_ports(self, msg, dpid, ingress):
        access_ports = set()
        # Include both sides of known inter-switch links so flooding stays on
        # host-facing ports and cannot form a broadcast cycle.
        for (src, dst), data in self.known_links.items():
            reverse = self.known_links.get((dst, src))
            access_ports.add((src, data["out_port"]))
            if reverse:
                access_ports.add((dst, reverse["out_port"]))
        for node, dp in self.datapaths.items():
            for port in dp.ports:
                port_no = int(port)
                if port_no > dp.ofproto.OFPP_MAX or (node, port_no) in access_ports:
                    continue
                if node == dpid and port_no == ingress:
                    continue
                dp.send_msg(dp.ofproto_parser.OFPPacketOut(
                    datapath=dp, buffer_id=dp.ofproto.OFP_NO_BUFFER,
                    in_port=dp.ofproto.OFPP_CONTROLLER,
                    actions=[dp.ofproto_parser.OFPActionOutput(port_no)], data=msg.data))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg, dp = ev.msg, ev.msg.datapath
        dpid, in_port = int(dp.id), int(msg.match["in_port"])
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None or eth.ethertype == 0x88CC:
            return
        interswitch = {(src, data["out_port"]) for (src, _), data in self.known_links.items()}
        if (dpid, in_port) not in interswitch:
            self.hosts[eth.src] = (dpid, in_port)
            arp_pkt = pkt.get_protocol(arp.arp)
            if arp_pkt:
                self.host_ips[arp_pkt.src_ip] = (dpid, in_port)
        arp_pkt = pkt.get_protocol(arp.arp)
        destination = self.hosts.get(eth.dst)
        if arp_pkt and arp_pkt.dst_ip in self.host_ips:
            destination = self.host_ips[arp_pkt.dst_ip]
        if destination is None:
            if arp_pkt:
                self._flood_arp_access_ports(msg, dpid, in_port)
            return
        source = self.hosts.get(eth.src)
        if source is None:
            return
        forward_result = self._build_path(source[0], destination[0])
        reverse_result = self._build_path(destination[0], source[0])
        if forward_result is None or reverse_result is None:
            return
        forward, forward_sources = forward_result
        reverse, reverse_sources = reverse_result
        forward, forward_sources = self._stabilize_route((eth.src, eth.dst), forward, forward_sources)
        reverse, reverse_sources = self._stabilize_route((eth.dst, eth.src), reverse, reverse_sources)
        ok = self._install_direction(forward, eth.src, eth.dst, destination)
        ok &= self._install_direction(reverse, eth.dst, eth.src, source)
        self.logger.info("flow_installation issued=%s forward=%s forward_sources=%s "
                         "reverse=%s reverse_sources=%s",
                         bool(ok), forward, forward_sources, reverse, reverse_sources)
        if not ok:
            return
        if dpid in forward and forward.index(dpid) < len(forward) - 1:
            out_port = self.graph[dpid][forward[forward.index(dpid) + 1]]["out_port"]
        elif dpid == destination[0]:
            out_port = destination[1]
        else:
            return
        dp.send_msg(dp.ofproto_parser.OFPPacketOut(
            datapath=dp, buffer_id=msg.buffer_id, in_port=in_port,
            actions=[dp.ofproto_parser.OFPActionOutput(int(out_port))],
            data=msg.data if msg.buffer_id == dp.ofproto.OFP_NO_BUFFER else None))
