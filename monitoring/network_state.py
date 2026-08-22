import time

class NetworkStateTracker:
    """
    Tracks port and link statistics over time, computing delta throughput (Mbps),
    capacity utilization, packet loss rates, and structuring the full network state.
    """

    def __init__(self, default_capacity_mbps=100.0, default_delay_ms=1.0):
        self.default_capacity_mbps = default_capacity_mbps
        self.default_delay_ms = default_delay_ms
        # Stores previous port counters: (dpid, port_no) -> {tx_bytes, rx_bytes, tx_packets, rx_packets, tx_errors, rx_dropped, timestamp}
        self.prev_port_stats = {}
        # Stores current calculated port metrics: (dpid, port_no) -> {throughput_mbps, utilization, loss_percent, ...}
        self.port_metrics = {}
        # Stores complete structured link state: (src_dpid, dst_dpid) -> dict
        self.link_metrics = {}

    def update_port_stats(self, dpid, port_no, tx_bytes, rx_bytes, tx_packets, rx_packets, tx_errors, rx_dropped):
        """
        Updates raw counters for a specific switch port and computes delta throughput, utilization, and packet loss.
        """
        now = time.time()
        key = (dpid, port_no)

        if key in self.prev_port_stats:
            prev = self.prev_port_stats[key]
            dt = now - prev['timestamp']

            if dt > 0:
                # Delta calculations
                delta_tx_bytes = max(0, tx_bytes - prev['tx_bytes'])
                delta_rx_bytes = max(0, rx_bytes - prev['rx_bytes'])
                total_bytes = delta_tx_bytes + delta_rx_bytes

                delta_tx_pkts = max(0, tx_packets - prev['tx_packets'])
                delta_rx_pkts = max(0, rx_packets - prev['rx_packets'])
                delta_errors = max(0, tx_errors - prev['tx_errors'])
                delta_drops = max(0, rx_dropped - prev['rx_dropped'])
                total_pkts = delta_tx_pkts + delta_rx_pkts + delta_errors + delta_drops

                # Throughput in Mbps: (bytes * 8) / (dt * 1e6)
                throughput_mbps = (total_bytes * 8.0) / (dt * 1e6)
                utilization = min(1.0, throughput_mbps / self.default_capacity_mbps)

                # Packet loss percentage
                loss_percent = (float(delta_errors + delta_drops) / float(total_pkts) * 100.0) if total_pkts > 0 else 0.0

                self.port_metrics[key] = {
                    "throughput_mbps": round(throughput_mbps, 4),
                    "utilization": round(utilization, 4),
                    "loss_percent": round(loss_percent, 4),
                    "total_bytes": tx_bytes + rx_bytes,
                    "timestamp": now
                }

        # Update previous stats
        self.prev_port_stats[key] = {
            "tx_bytes": tx_bytes,
            "rx_bytes": rx_bytes,
            "tx_packets": tx_packets,
            "rx_packets": rx_packets,
            "tx_errors": tx_errors,
            "rx_dropped": rx_dropped,
            "timestamp": now
        }

    def update_link_topology(self, topology_links):
        """
        Updates the link_metrics map using discovered topology links and calculated port metrics.
        topology_links is a list of Ryu Link objects (link.src, link.dst).
        """
        active_links = {}
        now = time.time()

        for link in topology_links:
            src_dpid = link.src.dpid
            dst_dpid = link.dst.dpid
            src_port = link.src.port_no
            dst_port = link.dst.port_no

            port_data = self.port_metrics.get((src_dpid, src_port), {
                "throughput_mbps": 0.0,
                "utilization": 0.0,
                "loss_percent": 0.0
            })

            link_key = (src_dpid, dst_dpid)
            active_links[link_key] = {
                "src_dpid": src_dpid,
                "dst_dpid": dst_dpid,
                "src_port": src_port,
                "dst_port": dst_port,
                "capacity_mbps": self.default_capacity_mbps,
                "throughput_mbps": port_data["throughput_mbps"],
                "utilization": port_data["utilization"],
                "delay_ms": self.default_delay_ms,
                "packet_loss_percent": port_data["loss_percent"],
                "status": "UP"
            }

        self.link_metrics = active_links

    def get_structured_state(self):
        """
        Returns a complete structured snapshot of the network state.
        """
        return {
            "timestamp": time.time(),
            "total_links": len(self.link_metrics),
            "links": {
                f"s{src}-s{dst}": metrics
                for (src, dst), metrics in self.link_metrics.items()
            }
        }
