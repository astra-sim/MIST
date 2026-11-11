
from .MIST_Coordinator import MISTCoordinator, CoordRouterType
from mist.Client import ClientType, MISTClient, LLMClient
from mist.Scheduler import SchedulerConfig, BatchingMethod
from mist.Input_requests import UniformDistribution, TraceDistributions, PoissonDistribution, LengthVariables
from mist.Clusters import ClusterConfig
from mist.Request import Request,RequestStage
from typing import List, Optional
from mist.Global_Network import get_network_bw_between_clients, get_network_spec
import pandas as pd
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

def client_to_stage_mapping(type: ClientType):
    if type == ClientType.RAG:
        return RequestStage.RAG
    elif type == ClientType.PREPROCESS:
        return RequestStage.PREPROCESS
    elif type == ClientType.PREFILL:
        return RequestStage.PREFILL
    elif type == ClientType.DECODE:
        return RequestStage.DECODE
    elif type == ClientType.POSTPROCESS:
        return RequestStage.POSTPROCESS
    else:
        raise ValueError(f"No stage  found for the client:{type}")

class MISTCoordinatorDisagg(MISTCoordinator):

    def __init__(
        self,
        request_queue_distr = None,
        model = 'meta-llama/meta-llama-3.1-70b',
        num_llm_clients=3,
        num_prefill_clients=1,
        num_decode_clients=1,
        num_mixed_clients=0,
        cluster = None,
        decode_cluster = None,
        cluster_schedule=CoordRouterType.JOIN_SHORTEST_QUEUE,
        convert_to_mixed_client=True,

        logging_file: Optional[str] = 'trace.json',
        max_sim_time: Optional[float] = np.inf,
        max_pending_prompt_tokens: int = 8192,
        ) -> None:

        # Defaults are built per instance: a request queue or cluster in the signature
        # would be created at import time and shared (and mutated) by every coordinator.
        if request_queue_distr is None:
            request_queue_distr = UniformDistribution(
                rps=30, sim_time=1000,
                input_vars=LengthVariables(4096,3), output_vars=LengthVariables(4096,3),
            ).request_queue
        if cluster is None:
            cluster = ClusterConfig(device='H100_GPU', tensor_parallel_size=4,
                                      model='meta-llama/meta-llama-3.1-70b', pipeline_parallel_size=1)

        assert num_prefill_clients != 0 and num_decode_clients != 0, "Must have initial prefill and decode clients"
        assert num_decode_clients + num_prefill_clients + num_mixed_clients == num_llm_clients, "Incorrect clients assignment"
        self.num_prefill_clients=num_prefill_clients
        self.num_decode_clients=num_decode_clients
        self.num_mixed_clients=num_mixed_clients

        self.cluster = cluster
        self.decode_cluster = decode_cluster if decode_cluster is not None else cluster
        self.model = model
        # Initialize the parent class
        super().__init__(
            starting_request_queue=request_queue_distr,
            logging_file=logging_file,
            max_sim_time=max_sim_time
            )

        # PARAMS for the cluster (system) configs
        self.num_llm_clients=num_llm_clients
        self.cluster_schedule=cluster_schedule

        # TODO: For Splitwise and DistServe, they should have this
        self.fixed_prefill_to_decode = True
        self.convert_to_mixed_client = convert_to_mixed_client

        # PARAMS for simulation logs
        self.req_cntr = 0
        self.sim_time = 1000
        self.current_time = 0

        # This is a queue used to track the movement and will be moved back to original state
        # value: (type moved from, type moved to, client_id)
        self.client_movement_tracker = deque()

        self.prefill_cntr=0
        self.decode_cntr=0
        self.mixed_cntr=0
        # Assign the clients to the
        self.client_matcher[ClientType.PREFILL] = []
        self.client_matcher[ClientType.DECODE] = []
        # Need a mixed pool for the purpose of easy management
        self.client_matcher[ClientType.MIXED] = []

        logger.debug(f"model used by cluster: {self.cluster.model}")

        pre_client_decode_only_cache = {}
        pre_client_mixed_batch_cache = {}
        for i in range(self.num_prefill_clients):
            prefill_client_to_add = LLMClient(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED, max_batched_kv_size=900000),
            cluster=self.cluster,
            client_id="PREFILL"+str(i),
            client_types = [ClientType.PREFILL, ClientType.DECODE],
            max_pending_prompt_tokens=max_pending_prompt_tokens,
            decode_only_cache = pre_client_decode_only_cache,
            mixed_batch_cache = pre_client_mixed_batch_cache,
            )
            self.add_client(prefill_client_to_add, [ClientType.PREFILL])
        
        dec_client_decode_only_cache = {}
        dec_client_mixed_batch_cache = {}
        for i in range(self.num_decode_clients):
            decode_client_to_add = LLMClient(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.MIXED),
            cluster=self.cluster,
            client_id="DECODE"+str(i),
            client_types = [ClientType.DECODE],
            max_pending_prompt_tokens=max_pending_prompt_tokens,
            decode_only_cache = dec_client_decode_only_cache,
            mixed_batch_cache = dec_client_mixed_batch_cache,
            
            )
            self.add_client(decode_client_to_add, [ClientType.DECODE])

        # Mixed clients are able to run both Prefill and Decode in chunked prefill manner
        for i in range(self.num_mixed_clients):
            mixed_client_to_add = LLMClient(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.CHUNKED, max_batched_kv_size=900000),
            cluster=self.cluster,
            client_id="MIXED"+str(i),
            client_types = [ClientType.PREFILL, ClientType.DECODE],
            max_pending_prompt_tokens=max_pending_prompt_tokens,
            decode_only_cache = pre_client_decode_only_cache,
            mixed_batch_cache = pre_client_mixed_batch_cache,
            )
            self.add_client(mixed_client_to_add, [ClientType.MIXED])

        # # Add An Client for the HOST (Determine characteristics of host later)
        # self.add_client(MISTClient(
        #     model = self.model,
        #     sim_duration = 10000000,
        #     client_types = [ClientType.HOST],
        #     ), [ClientType.HOST])

        # Print out layout and assignment of clients
        logger.info(" ".join([
            f"Prefill Clients: {self.client_matcher[ClientType.PREFILL]},",
            f"Decode Clients: {self.client_matcher[ClientType.DECODE]},",
            f"Mixed Clients: {self.client_matcher[ClientType.MIXED]}"
        ]))

    def _check_the_client_load(self, client_id):
        assert client_id < self.num_llm_clients, f"Invalid Client id"
        client_load_num = len(self.clients[client_id].request_queue)

    def _reassign_client_type(self, client_id, old_type:ClientType, new_type:ClientType):
        """
            TODO: This is used by Splitwise-like system to reassign client to another type
        """
        assert client_id < self.num_llm_clients, f"Invalid Client id"
        assert len(self.client_matcher[old_type]) > 0, f"cannot remove more clients from type -- {old_type}"
        self.client_matcher[old_type].remove(client_id)
        self.client_matcher[new_type].append(client_id)
        # print(f"reassigning client: {client_id} from {old_type} to {new_type}")
        if old_type == ClientType.MIXED:
            # Move back to the original client requires remove the capability of running the opposite types
            self.clients[client_id].client_types.remove(self._get_opposite_client_type(new_type))
            self.clients[client_id].scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED)
        else:
        # Capable to run both prefill and decode now
            self.clients[client_id].client_types.append(self._get_opposite_client_type(old_type))
            self.clients[client_id].scheduler_config = SchedulerConfig(batching_method=BatchingMethod.MIXED)

    def _get_least_loaded_clients(self, client_types:List[ClientType], request:Request):
        """
            Return the least loaded clients for a list of client types inside the coordinator
        """
        client_to_select = -1
        # print(f"Client Type:{client_type} number of clients: {num_clients}")
        client_loads = {}
        for eng_type in client_types:
            num_clients = len(self.client_matcher[eng_type])
            for type_id in range(num_clients):
                client_id = self.client_matcher[eng_type][type_id]
                client_loads[(client_id, eng_type)] = self.clients[client_id].tokens_load(request.metrics.arrival_time)

        ## If all the clients have the same load, just return in RR manner for the original type (prefill or decode)
        if len(set(client_loads.values())) == 1:
            return super()._determine_dst_client(client_types[0])
        ## Else find the least loaded client
        elif len(client_loads) > 0:
            (client_to_select, eng_type) = min(client_loads, key=client_loads.get)

        return client_to_select

    def _get_least_loaded_clients_prefill(self, client_types:List[ClientType], request:Request):
        """
            Return the least loaded clients for a list of client types inside the coordinator
        """
        client_to_select = -1
        # print(f"Client Type:{client_type} number of clients: {num_clients}")
        client_loads = {}
        for eng_type in client_types:
            num_clients = len(self.client_matcher[eng_type])
            for type_id in range(num_clients):
                client_id = self.client_matcher[eng_type][type_id]
                client_loads[(client_id,eng_type)] = self.clients[client_id].tokens_load(request.metrics.arrival_time) + request.input_len

        ## If all the clients have the same load, just return in RR manner for the original type (prefill or decode)
        if len(set(client_loads.values())) == 1 and len(client_loads) > 0:
            client_to_select = super()._determine_dst_client(client_types[0])
        elif len(client_loads) > 0:
            (client_to_select, eng_type) = min(client_loads, key=client_loads.get)
        return client_to_select

    def _get_least_loaded_client(self, client_type:ClientType, request:Request):
        """
            Return the least loaded client for a single client type inside the coordinator
        """
        client_to_select = -1
        num_clients = len(self.client_matcher[client_type])
        # print(f"Client Type:{client_type} number of clients: {num_clients}")
        client_loads = {}
        for type_id in range(num_clients):
            client_id = self.client_matcher[client_type][type_id]
            client_loads[client_id] = self.clients[client_id].tokens_load(request.metrics.arrival_time)
        # print("Client Loads: ", client_loads)
        ## If all the clients have the same load, just return in RR manner
        if len(set(client_loads.values())) == 1 or sum(1 for load in client_loads.values() if load == 0) > 0:
            return super()._determine_dst_client(client_type)
        ## Else find the least loaded client
        elif len(client_loads) > 0:
            client_to_select = min(client_loads, key=client_loads.get)
        return client_to_select

    # TODO: This can be removed, because add input length of request does not change anything
    def _get_least_loaded_client_prefill(self, client_type:ClientType, request:Request):
        """
            Return the least loaded client for a single client type inside the coordinator
        """
        client_to_select = -1
        num_clients = len(self.client_matcher[client_type])
        # print(f"Client Type:{client_type} number of clients: {num_clients}")
        client_loads = {}
        for type_id in range(num_clients):
            client_id = self.client_matcher[client_type][type_id]
            client_loads[client_id] = self.clients[client_id].tokens_load(self.global_time) + request.input_len

        ## If all the clients have the same load, just return in RR manner
        if len(set(client_loads.values())) == 1:
            return super()._determine_dst_client(client_type)
        ## Else find the least loaded client
        elif len(client_loads) > 0:
            client_to_select = min(client_loads, key=client_loads.get)
        return client_to_select

    def _get_opposite_client_type(self, client_type:ClientType):
        assert client_type in [ClientType.PREFILL, ClientType.DECODE], "it has to be either PREFILL or DECODE"
        if client_type == ClientType.PREFILL:
            return ClientType.DECODE
        else:
            return ClientType.PREFILL

    def _get_mixed_client_overloaded(self, init_client_type:ClientType, request:Request):
        # Go over all mixed clients to check if there is a good loaded client
        client_to_select = self._get_least_loaded_client(ClientType.MIXED, request)
        # print(f"number of mixed client {len(self.client_matcher[ClientType.MIXED])}")

        if client_to_select == -1 or self.clients[client_to_select].check_is_overload(request.metrics.arrival_time):
            if self.convert_to_mixed_client:
                # If mixed clients are overloaded
                type_to_find = self._get_opposite_client_type(init_client_type)
                client_to_select = self._get_least_loaded_client(type_to_find, request)
                # print(f"No Mixed found, converting {type_to_find} to mixed client. Client to select: {client_to_select}")
                if len(self.client_matcher[type_to_find]) < 1 or self.clients[client_to_select].check_is_overload(request.metrics.arrival_time):
                    # All the clients are overloaded, just don't move, find the
                    # least loaded client in the original pool
                    client_to_select = self._get_least_loaded_client(init_client_type, request)
                else:
                    self._reassign_client_type(client_to_select, type_to_find, ClientType.MIXED)

                    # print(f"Convert {type_to_find} to {ClientType.MIXED} at time: {self.global_time}")
                    # Track the movement that just took place
                    self.client_movement_tracker.append((type_to_find, ClientType.MIXED, client_to_select))
            else:
                # Not convert to mixed client, Find the client that has the least load in both initiated client pool or Mixed client pool
                client_to_select = self._get_least_loaded_clients([init_client_type, ClientType.MIXED], request)
                assert client_to_select != -1, f"must select an client because no movement of clients between pools"
        
        return client_to_select

    def _get_mixed_client_overloaded_prefill(self, init_client_type:ClientType, request:Request=None):
        # Go over all mixed clients to check if there is a good loaded client
        client_to_select = self._get_least_loaded_client_prefill(ClientType.MIXED, request)
        # print(f"number of mixed client {len(self.client_matcher[ClientType.MIXED])}")

        if client_to_select == -1 or self.clients[client_to_select].check_is_overload_with_prefill(request.metrics.arrival_time, request):
            if self.convert_to_mixed_client:
                # If mixed clients are overloaded
                type_to_find = self._get_opposite_client_type(init_client_type)
                client_to_select = self._get_least_loaded_client_prefill(type_to_find, request)
                # print(f"No Mixed found, converting {type_to_find} to mixed client. Client to select: {client_to_select}")
                if len(self.client_matcher[type_to_find]) < 1 or self.clients[client_to_select].check_is_overload_with_prefill(request.metrics.arrival_time, request):
                    # All the clients are overloaded, just don't move, find the
                    # least loaded client in the original pool
                    client_to_select = self._get_least_loaded_clients_prefill([init_client_type, ClientType.MIXED], request)
                else:
                    self._reassign_client_type(client_to_select, type_to_find, ClientType.MIXED)

                    # print(f"Convert {type_to_find} to {ClientType.MIXED} at time: {self.global_time}")
                    # Track the movement that just took place
                    self.client_movement_tracker.append((type_to_find, ClientType.MIXED, client_to_select))
            else:
                # Not convert to mixed client, Find the client that has the least load in both initiated client pool or Mixed client pool
                client_to_select = self._get_least_loaded_clients([init_client_type, ClientType.MIXED], request)
                assert client_to_select != -1, f"must select an client because no movement of clients between pools"
        
        return client_to_select

    def _determine_prefill_client(self, request:Request):
        assert self.num_prefill_clients > 0, "Cannot run with 0 prefill clients"
        if self.cluster_schedule == CoordRouterType.ROUND_ROBIN:
            #TODO: Use RoundRobin in base class, may need to do the client movement like JSQ later.
            return super()._determine_dst_client(ClientType.PREFILL)
        elif self.cluster_schedule == CoordRouterType.JOIN_SHORTEST_QUEUE:
            client_to_select = self._get_least_loaded_client(ClientType.PREFILL, request)
            # print(f'Req:{request.request_id} client to select:', client_to_select)
            if client_to_select != -1 and self.clients[client_to_select].check_is_overload_with_prefill(request.metrics.arrival_time, request):
                # If the client is overloaded, we need to use clients in the mixed pool
                client_to_select = self._get_mixed_client_overloaded_prefill(ClientType.PREFILL,request)

            if client_to_select == -1:
                # If the client to select is still -1, go to find Mixed Client
                client_to_select = self._get_least_loaded_client(ClientType.MIXED, request)

            if self.fixed_prefill_to_decode:
                # Determine the decode client in here
                request.client_assigned_stage[RequestStage.DECODE] = self._determine_decode_client(request)
                ## Keeping track of the request that is enqueued to the decode client
            return client_to_select

        else:
            raise ValueError(f"Coordinator Scheduler Type Not Supported:{self.cluster_schedule}")

    # Use Join Shortest Queue below, because RoundRobin is used in the base class
    def _determine_decode_client(self, request:Request):
        assert self.num_prefill_clients > 0, "Cannot run with 0 prefill clients"
        if self.cluster_schedule == CoordRouterType.ROUND_ROBIN:
            #TODO: Do a similar thing like Prefill
            return super()._determine_dst_client(ClientType.PREFILL)
        elif self.cluster_schedule == CoordRouterType.JOIN_SHORTEST_QUEUE:
            client_to_select = self._get_least_loaded_client(ClientType.DECODE, request)
            # print(f"client id: {self.clients[client_to_select].client_id}, type: {self.clients[client_to_select].client_types}")
            if client_to_select != -1 and self.clients[client_to_select].check_is_overload(request.metrics.arrival_time):
                # If the client is overloaded, we need to use clients in the mixed pool
                client_to_select = self._get_mixed_client_overloaded(ClientType.DECODE, request)

            if client_to_select == -1:
                # If the client to select is still -1, go to find
                client_to_select = self._get_least_loaded_client(ClientType.MIXED, request)
            return client_to_select
        else:
            raise ValueError(f"Coordinator Scheduler Type Not Supported:{self.cluster_schedule}")

    # def _determine_mixed_client(self, request:Request):
    #     mixed_client_id = self.mixed_cntr % self.num_prefill_clients
    #     self.mixed_cntr += 1
    #     client_id = self.client_matcher[ClientType.MIXED][mixed_client_id]
    #     return client_id

    # TODO: Splitwise has optimized movement protocal that does per-layer transfer
    def _get_move_request_time(self, request, src_client, dst_client) -> float:
        # Get the cost of moving the request from src to dst client
        return super()._get_move_request_time(request, src_client, dst_client)
    def initial_client_connection_df(self):
        network_spec = get_network_spec("H100_nvl8")
        intra_domain_sz = self.cluster.tensor_parallel_size * self.cluster.pipeline_parallel_size
        global_size = self.num_llm_clients
        self.client_connection_bw_latency_df = get_network_bw_between_clients(global_size, network_spec, intra_domain_sz)
        ## Each node has 1 CPU host
        for i in range(self.num_total_clients):
            for j in range(self.num_llm_clients, self.num_total_clients):
                bw = network_spec.intra_node_bw * network_spec.intra_node_eff
                latency = network_spec.intra_node_latency
                # Append bidirectional connections
                cur_data = [
                    {"src": i, "dst": j, "latency(msec)": latency, "BW(GB/s)": bw},
                    {"src": j, "dst": i, "latency(msec)": latency, "BW(GB/s)": bw},
                ]
                df = pd.DataFrame(cur_data)
                self.client_connection_bw_latency_df = pd.concat([self.client_connection_bw_latency_df, df])

    def _determine_dst_client(self, client_type, request):
        """
            Overloaded function to determine the destination client
        """
        # TODO: Temporarily put the determination of the client movement here and also need to see
        # if the determination is too frequent
        pop_count = len(self.client_movement_tracker)
        while self.client_movement_tracker and pop_count > 0:
            orig_type, cur_type, client_id = self.client_movement_tracker.popleft()  # Pop from front
            pop_count -= 1
            eng_type = self._get_opposite_client_type(orig_type)
            req_stage = client_to_stage_mapping(eng_type)
            if not self.clients[client_id].check_req_stage_existence(req_stage):
                self._reassign_client_type(client_id, cur_type, orig_type)
            else:
                self.client_movement_tracker.append((orig_type, cur_type, client_id))

        # TODO: May combine PEFILL into the same function (the callee functions are similar)
        if client_type == ClientType.PREFILL:
            prefill_client = self._determine_prefill_client(request)
            # print(f"determine dest for req: {request.request_id}, with client: {client_type} assigned to {prefill_client}")
            return prefill_client
        elif client_type == ClientType.DECODE:
            if self.fixed_prefill_to_decode:
                # If the decode client is determined while prefill
                assert request.client_assigned_stage[RequestStage.DECODE] != None, "The decode should be assigned"
                decode_client = request.client_assigned_stage[RequestStage.DECODE]
            else:
                decode_client = self._determine_decode_client(request)
            # print(f"determine dest for req: {request.request_id}, with client: {client_type} assigned to {decode_client}")
            return decode_client
        else:
            return super()._determine_dst_client(client_type)