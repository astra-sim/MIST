import os
import tempfile
from copy import deepcopy

# Fresh latency cache per test session so results never come from a stale disk cache.
os.environ["MIST_CACHE_DIR"] = tempfile.mkdtemp(prefix="mist-test-cache-")

import pytest

from mist import BatchingMethod, LengthVariables, ClusterConfig, PoissonDistribution, SchedulerConfig
from mist.Coordinator import MISTCoordinator
from mist.Client import ClientType, LLMClient

MODEL = "meta-llama/meta-llama-3.1-8b"


def make_queue(seed=0, rps=4, sim_time=3000):
    return PoissonDistribution(
        rps=rps, sim_time=sim_time,
        input_vars=LengthVariables(512, 128), output_vars=LengthVariables(64, 16),
        rand_seed=seed,
    ).request_queue


def run_sim(queue, device="H100_GPU", batching=BatchingMethod.CHUNKED, **sched_kwargs):
    coord = MISTCoordinator(deepcopy(queue), logging_file=None)
    types = [ClientType.PREFILL, ClientType.DECODE]
    client = LLMClient(
        model=MODEL, client_types=types,
        scheduler_config=SchedulerConfig(batching_method=batching, chunk_size=512, **sched_kwargs),
        cluster=ClusterConfig(device=device, tensor_parallel_size=1, model=MODEL),
    )
    coord.add_client(client, types)
    coord.run_sim()
    return coord


@pytest.fixture
def queue():
    return make_queue()
