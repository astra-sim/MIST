"""ASTRA-sim backend wiring in the coordinator.

A stand-in for the compiled `simulation_py_congestion_aware` module models each
transfer as latency + size / bandwidth with no congestion. With the same link
parameters, the ASTRA-sim path must then reproduce the analytical link model.
"""
import heapq
import sys
import types
from copy import deepcopy

import pandas as pd
import pytest

from conftest import make_queue
from mist import ClusterConfig, Request
from mist.Coordinator import MISTCoordinatorDisagg

MODEL = "meta-llama/meta-llama-3.1-8b"
LATENCY_MS, BW_GBPS = 0.5, 25


def _fake_backend():
    class EventQueue:
        def __init__(self):
            self.now, self.events, self.arrived = 0, [], []

        def get_current_time(self):
            return self.now

        def finished(self):
            return not self.events

        def proceed(self):
            self.now, _, req_id = heapq.heappop(self.events)
            self.arrived.append(req_id)

        def get_next_event_time(self):
            return self.events[0][0]

        def advance_to(self, t):
            assert t >= self.now and (not self.events or t < self.events[0][0])
            self.now = t

        def get_and_clear_arrivals(self):
            out, self.arrived = self.arrived, []
            return out

    class Topology:
        def get_npus_count(self):
            return 4

        def send_python(self, chunk):
            size, req_id, q = chunk
            # Same units as MISTCoordinator._get_move_request_time: MB / (GB/s) = ms.
            ms = LATENCY_MS + size / 2**20 / BW_GBPS
            heapq.heappush(q.events, (q.now + round(ms * 1e6), req_id, req_id))

        @staticmethod
        def set_event_queue(q):
            pass

    class Chunk:
        @staticmethod
        def create_with_event_queue(size, src, dst, req_id, topology, event_queue):
            return (size, req_id, event_queue)

    m = types.ModuleType("simulation_py_congestion_aware")
    m.EventQueue, m.Topology, m.Chunk = EventQueue, Topology, Chunk
    m.NetworkParser = lambda path: path
    m.construct_topology = lambda parser: Topology()
    return m


def _run(queue, **kw):
    coord = MISTCoordinatorDisagg(
        request_queue_distr=deepcopy(queue), model=MODEL,
        cluster=ClusterConfig(device="H100_GPU", tensor_parallel_size=1, model=MODEL),
        num_llm_clients=2, num_prefill_clients=1, num_decode_clients=1,
        logging_file=None, **kw)
    coord.network_file = pd.DataFrame(
        [[s, d, 0 if s == d else LATENCY_MS, 0 if s == d else BW_GBPS] for s in range(2) for d in range(2)],
        columns=["src", "dst", "latency(msec)", "BW(GB/s)"])
    coord.run_sim()
    return coord


def test_astrasim_path_matches_analytical_links(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "simulation_py_congestion_aware", _fake_backend())
    topo = tmp_path / "topo.yml"
    topo.write_text("")
    queue = make_queue(rps=4, sim_time=2000)

    ref = _run(queue)
    net = _run(queue, astrasim_topology=str(topo))

    assert net.request_serviced == ref.request_serviced == len(queue)
    assert not net._in_flight and net.network.next_event_ms() is None
    kv = lambda c: sorted((r.request_id, r.metrics.kv_transfer_time) for r in c.completed_requests)
    for (i, a), (j, b) in zip(kv(ref), kv(net)):
        assert i == j and b == pytest.approx(a, abs=1e-5) and b > 0
    assert net.get_global_stats().TTFT == pytest.approx(ref.get_global_stats().TTFT, abs=1e-5)


def test_missing_topology_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _run(make_queue(rps=1, sim_time=500), astrasim_topology=str(tmp_path / "nope.yml"))


def test_finite_run_counts_prefill_while_kv_is_in_transit(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "simulation_py_congestion_aware", _fake_backend())
    monkeypatch.setattr(ClusterConfig, "get_chunked_time", lambda self, prefills, decodes: (10, 0))
    topology = tmp_path / "topo.yml"
    topology.write_text("")

    coord = _run(
        [Request(input_len=128, output_len=8)],
        astrasim_topology=str(topology),
        max_sim_time=10.25,
    )
    stats = coord.get_global_stats()

    assert set(coord._in_flight) == {0}
    assert coord.request_serviced == 0
    assert stats.output_throughput == pytest.approx(1 / 0.01)
    assert stats.total_token_throughput == pytest.approx(129 / 0.01)


def test_clocks_stay_aligned_under_contention(tmp_path, monkeypatch):
    """One FIFO link: a transfer sent while another is in flight waits for it, and no
    transfer finishes before it was sent or before the previous one on the link."""
    backend = _fake_backend()
    link_free = [0]
    sends, sizes = [], {}

    def send_python(self, chunk):
        size, req_id, q = chunk
        start = max(q.now, link_free[0])
        link_free[0] = start + round((LATENCY_MS + size / 2**20 / BW_GBPS) * 1e6)
        sends.append((req_id, q.now, link_free[0]))
        sizes[req_id] = size
        heapq.heappush(q.events, (link_free[0], req_id, req_id))

    backend.Topology.send_python = send_python
    monkeypatch.setitem(sys.modules, "simulation_py_congestion_aware", backend)
    topo = tmp_path / "topo.yml"
    topo.write_text("")
    queue = make_queue(rps=40, sim_time=2000)
    net = _run(queue, astrasim_topology=str(topo))

    assert net.request_serviced == len(queue)
    alone = {req_id: round((LATENCY_MS + size / 2**20 / BW_GBPS) * 1e6) for req_id, size in sizes.items()}
    assert any(end - sent > 1.5 * alone[req_id] for req_id, sent, end in sends), "expected queueing on the link"
    by_id = {r.request_id: r for r in net.completed_requests}
    for req_id, sent_ns, end_ns in sends:
        kv = by_id[req_id].metrics.kv_transfer_time
        if kv is not None:
            assert kv == pytest.approx((end_ns - sent_ns) / 1e6, abs=1e-5)


def test_transfer_finishing_at_batch_start_joins_that_batch():
    """At equal times, a finished transfer's STAGE_PUSH runs before the client's step."""
    from mist.Coordinator.MIST_Coordinator import EventType
    order = sorted([EventType.CLIENT_STEP, EventType.NETWORK_SEND, EventType.NETWORK_STEP, EventType.STAGE_PUSH])
    assert order == [EventType.STAGE_PUSH, EventType.NETWORK_STEP, EventType.NETWORK_SEND, EventType.CLIENT_STEP]
