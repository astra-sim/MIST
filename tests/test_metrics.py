import math

import pytest

from conftest import make_queue, run_sim
from mist import ClusterConfig, Request


def test_end_to_end_metrics_are_sane(queue):
    coord = run_sim(queue)
    stats = coord.get_global_stats()
    assert coord.request_serviced == len(queue) > 0
    assert 0 < stats.TTFT < stats.T99_latency
    assert stats.TPOT > 0
    assert stats.T50_latency <= stats.T90_latency <= stats.T99_latency
    assert stats.total_token_throughput > stats.output_throughput > 0


def test_total_throughput_ignores_reused_kv():
    """past_context (reused prefix KV) far larger than input_len must not drive throughput negative."""
    plain = make_queue()
    with_kv = make_queue()
    for req in with_kv:
        req.past_context = 20 * req.input_len
    stats = run_sim(with_kv).get_global_stats()
    assert stats.total_token_throughput > stats.output_throughput > 0
    # Prefilled tokens per request are the same with or without reused KV.
    base = run_sim(plain).get_global_stats()
    prompt_tokens_per_req = lambda s: (s.total_token_throughput - s.output_throughput) / s.rps
    assert math.isclose(prompt_tokens_per_req(stats), prompt_tokens_per_req(base))

@pytest.fixture
def fixed_batch_time(monkeypatch):
    monkeypatch.setattr(ClusterConfig, "get_chunked_time", lambda self, prefills, decodes: (10, 0))


def test_finite_run_counts_started_work_through_its_completion(fixed_batch_time):
    coord = run_sim(
        [
            Request(arrival_time=0, input_len=128, output_len=2),
            Request(arrival_time=30, input_len=128, output_len=8),
        ],
        max_sim_time=35,
    )
    first, second = coord.starting_request_queue
    stats = coord.get_global_stats()

    assert [req.request_id for req in coord.completed_requests] == [first.request_id]
    assert second.gen_tokens == 1
    assert second.data[0].finished_time == 40
    assert stats.rps == pytest.approx(1 / 0.04)
    assert stats.output_throughput == pytest.approx(3 / 0.04)
    assert stats.total_token_throughput == pytest.approx((2 * 128 + 3) / 0.04)


@pytest.mark.parametrize("input_len,prefilled,generated", [(128, 128, 1), (600, 512, 0)])
def test_finite_run_reports_work_before_any_request_finishes(
    fixed_batch_time, input_len, prefilled, generated
):
    coord = run_sim([Request(arrival_time=0, input_len=input_len, output_len=8)], max_sim_time=5)
    stats = coord.get_global_stats()

    assert not coord.completed_requests
    assert stats.rps == stats.T99_latency == 0
    assert stats.output_throughput == pytest.approx(generated / 0.01)
    assert stats.total_token_throughput == pytest.approx((prefilled + generated) / 0.01)


def test_finite_run_excludes_events_at_the_cutoff(fixed_batch_time):
    coord = run_sim([Request(arrival_time=10, input_len=128, output_len=1)], max_sim_time=10)
    req = coord.starting_request_queue[0]

    assert req.gen_tokens == 0
    assert all(stage.client_entry_time is None for stage in req.stage_metrics.values())
    assert not coord.completed_requests
    assert coord.get_global_stats().total_token_throughput == 0
