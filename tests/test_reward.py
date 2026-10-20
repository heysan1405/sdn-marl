from simulator.reward import RewardCalculator
import networkx as nx
from simulator.network_simulator import NetworkSimulator


def test_reward_throughput_uses_original_demand_after_rate_limiting():
    calculator = RewardCalculator()
    components = calculator.calculate_components({
        "total_requested_traffic_mbps": 70.0,
        "total_original_requested_traffic_mbps": 100.0,
        "total_throughput_mbps": 70.0,
        "congestion_ratio": 0.0,
        "average_path_delay_ms": 1.0,
        "average_packet_loss": 0.0,
        "failed_links": [], "failed_nodes": [],
        "link_statistics": [{}], "node_statistics": [{}],
    })
    assert components["throughput"] == 0.7


def test_old_metric_results_keep_legacy_requested_volume_fallback():
    calculator = RewardCalculator()
    components = calculator.calculate_components({
        "total_requested_traffic_mbps": 100.0,
        "total_throughput_mbps": 80.0,
        "congestion_ratio": 0.0,
        "average_path_delay_ms": 1.0,
        "average_packet_loss": 0.0,
        "failed_links": [], "failed_nodes": [],
        "link_statistics": [{}], "node_statistics": [{}],
    })
    assert components["throughput"] == 0.8


def test_isolating_an_undelivered_failed_flow_does_not_hide_its_loss():
    graph = nx.Graph([(0, 1), (1, 2), (0, 2)])
    simulator = NetworkSimulator(graph)
    simulator.demands = [{"id": 0, "src": 0, "dst": 2, "traffic_mbps": 100.0}]
    simulator.original_traffic = {0: 100.0}
    simulator.active_paths = {0: [0, 1, 2]}
    simulator.fail_link(0, 1)

    before = simulator.calculate_network_metrics()
    simulator.isolate_failed_components()
    after = simulator.calculate_network_metrics()

    assert before["average_packet_loss"] == 1.0
    assert after["average_packet_loss"] == 1.0
    assert before["delivered_demands"] == after["delivered_demands"] == 0
