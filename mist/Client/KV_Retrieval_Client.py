from .MIST_Client import MISTClient, ClientMetrics, ClientType
from mist.Scheduler.scheduler import Scheduler, SchedulerConfig, BatchingMethod
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
from mist.Clusters import MemoryCacheConfig
from mist.Request import Request, DataMetrics, RequestMetrics, RequestStage
import logging
logger = logging.getLogger(__name__)
import heapq
import time
import math
import time
import random

class KVRetrievalClient(MISTClient):

    def __init__(
        self,
        model: str,
        client_id:str=None,
        cluster: MemoryCacheConfig = None,
        client_types: List[ClientType] = [ClientType.CACHE_RETRIEVAL],
        ) -> None:

        super().__init__(
            model=model,
            client_id=client_id,
            cluster=cluster,
            client_types=client_types,
        )

    def get_machine_runtimes(self, requests: List[Request]) -> List[float]:
        """
            Get the runtime for each request in the current cycle.

            Inputs:
            requests : List of all the requests in the current cycle.

            Outputs:
            runtimes: List of runtimes for each request.
        """
        req = requests[0]
        # print(f"Getting Machine Runtimes for req:{req.request_id}, {req.input_len}, {req.output_len}, {req.past_context}")
        retrieval_time = self.cluster.get_KV_cache_time(requests)
        return retrieval_time