
from .MIST_Coordinator import MISTCoordinator, CoordRouterType
from mist.Client import ClientType, MISTClient, LLMClient
from mist.Scheduler import SchedulerConfig, BatchingMethod
from mist.Input_requests import UniformDistribution, TraceDistributions, PoissonDistribution, LengthVariables
from mist.Clusters import ClusterConfig
from mist.Request import Request,RequestStage
from typing import List
from mist.Global_Network import get_network_bw_between_clients, get_network_spec
import pandas as pd
from collections import deque
import math
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

class MISTCoordinatorDisagg_DistServe(MISTCoordinator):

    def __init__(
        self,
        request_queue_distr = None,
        model = 'meta-llama/meta-llama-3.1-70b',
        num_llm_clients=2,
        num_prefill_clients=1,
        num_decode_clients=1,
        cluster = None,
        cluster_schedule=CoordRouterType.JOIN_SHORTEST_QUEUE
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

        assert num_decode_clients + num_prefill_clients == num_llm_clients, "Incorrect clients assignment"
        self.num_prefill_clients=num_prefill_clients
        self.num_decode_clients=num_decode_clients
        self.num_llm_clients=num_llm_clients

        self.cluster = cluster
        self.model = model
        # Initialize the parent class
        super().__init__(request_queue_distr)

        # PARAMS for the cluster (system) configs
        self.cluster_schedule=cluster_schedule
        # PARAMS for simulation logs
        self.req_cntr = 0
        self.sim_time = 1000
        self.current_time = 0

       

        # This is a queue used to track the movement and will be moved back to original state
        # value: (type moved from, type moved to, client_id)
        self.client_movement_tracker = deque()
        
        # Used to help identify the group of decodes to be assigned to
        self.prefill_to_decode_matcher = dict()

        self.prefill_cntr=0
        self.decode_cntr=0

         # TODO: specify the affinity in here to choose either add_client/place_client, use post_init for sugar
        # Assign the clients to the 
        self.client_matcher[ClientType.PREFILL] = []
        self.client_matcher[ClientType.DECODE] = []
        for i in range(self.num_prefill_clients):
            prefill_client_to_add = LLMClient(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED),
            cluster=self.cluster,
            client_id="PREFILL"+str(i),
            client_types = [ClientType.PREFILL],
            )
            self.add_client(prefill_client_to_add, [ClientType.PREFILL])
        for i in range(self.num_decode_clients):
            decode_client_to_add = LLMClient(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED),
            cluster=self.cluster,
            client_id="DECODE"+str(i),
            client_types = [ClientType.DECODE],
            )
            self.add_client(decode_client_to_add, [ClientType.DECODE])

        # # Add An Client for the HOST (Determine characteristics of host later)
        # self.add_client(MISTClient(
        #     model = self.model,
        #     sim_duration = 10000000,
        #     client_types = [ClientType.HOST],
        #     ), [ClientType.HOST])

        # Print out layout and assignment of clients
        logger.info(" ".join([
            f"Decode Clients: {len(self.client_matcher[ClientType.PREFILL])},",
            f"Prefill Clients: {len(self.client_matcher[ClientType.DECODE])},"
        ]))

    def _check_the_client_load(self, client_id):
        assert client_id < self.num_llm_clients, f"Invalid Client id"
        client_load_num = len(self.clients[client_id].request_queue)

    def _get_least_loaded_client(self, client_list:List, request:Request):
        client_to_select = -1
        # print(f"Client Type:{client_type} number of clients: {num_clients}")
        client_loads = {}
        # for type_id in range(num_clients):
        #     client_id = self.client_matcher[client_type][type_id]
        #     client_loads[client_id] = self.clients[client_id].tokens_load(self.global_time)
        for client_id in client_list:
            logger.debug(f"At time: {request.metrics.arrival_time}, client: {client_id} has load: {self.clients[client_id].tokens_load(request.metrics.arrival_time)}")
            client_loads[client_id] = self.clients[client_id].tokens_load(request.metrics.arrival_time)

        ## If all the clients have the same load, just return in RR manner
        if len(set(client_loads.values())) == 1:
            client_to_select= client_list.pop(0)
            client_list.append(client_to_select)
        ## Else find the least loaded client
        elif len(client_loads) > 0:
            client_to_select = min(client_loads, key=client_loads.get)
        
        logger.debug(f"At time: {request.metrics.arrival_time}, {client_to_select} is selected")
        return client_to_select
    
    # def _get_least_loaded_client_prefill(self, client_type:ClientType):
    #     client_to_select = -1
    #     num_clients = len(self.client_matcher[client_type])
    #     # print(f"Client Type:{client_type} number of clients: {num_clients}")
    #     client_loads = {}
    #     for type_id in range(num_clients):
    #         client_id = self.client_matcher[client_type][type_id]
    #         client_loads[client_id] = self.clients[client_id].tokens_load(self.global_time)

    #     ## If all the clients have the same load, just return in RR manner
    #     if len(set(client_loads.values())) == 1:
    #         return super()._determine_dst_client(client_type, None)
    #     ## Else find the least loaded client
    #     elif len(client_loads) > 0:
    #         client_to_select = min(client_loads, key=client_loads.get)
    #     return client_to_select


    def _determine_prefill_client(self, request:Request):
        assert self.num_prefill_clients > 0, "Cannot run with 0 prefill clients"
        if self.cluster_schedule == CoordRouterType.ROUND_ROBIN:
            #TODO: Use RoundRobin in base class, may need to do the client movement like JSQ later.
            return super()._determine_dst_client(ClientType.PREFILL, request)
        elif self.cluster_schedule == CoordRouterType.JOIN_SHORTEST_QUEUE:
            logger.debug(f"request: {request.request_id} find clients: {self.client_matcher[ClientType.PREFILL]}")
            client_to_select = self._get_least_loaded_client(self.client_matcher[ClientType.PREFILL], request)
            request.client_assigned_stage[RequestStage.DECODE] = self._determine_decode_client(request, client_to_select)
            return client_to_select
        else:
            raise ValueError(f"Coordinator Scheduler Type Not Supported:{self.cluster_schedule}")

    # Use Join Shortest Queue below, because RoundRobin is used in the base class
    def _determine_decode_client(self, request:Request, prefill_id=-1):
        assert self.num_prefill_clients > 0, "Cannot run with 0 prefill clients"
        if self.cluster_schedule == CoordRouterType.ROUND_ROBIN:
            #TODO: Do a similar thing like Prefill
            return super()._determine_dst_client(ClientType.PREFILL, request)
        elif self.cluster_schedule == CoordRouterType.JOIN_SHORTEST_QUEUE:
            if request.client_assigned_stage[RequestStage.DECODE] == None:
                client_to_select = self._get_least_loaded_client(self.prefill_to_decode_matcher[prefill_id], request)
            else:
                client_to_select = request.client_assigned_stage[RequestStage.DECODE]
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
        data_size = request.get_data_movement_size(self.clients[dst_client].model)/2**20   # Convert to MB
        if src_client == dst_client:
            return 0
        else:
            latency, BW = self.compute_bw_latency(self.client_connection_bw_latency_df, src_client, dst_client)
            return latency + data_size/BW

    def intial_client_connection_df(self):
        network_spec = get_network_spec("H100_nvl8")
        intra_domain_sz = self.cluster.tensor_parallel_size * self.cluster.pipeline_parallel_size
        global_size = self.num_llm_clients
        self.client_connection_bw_latency_df = get_network_bw_between_clients(global_size, network_spec, intra_domain_sz)
        ## Each node has 1 CPU host
        for i in range(self.num_llm_clients):
            j = self.num_llm_clients
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

        # print(f"determine dest for req: {request.request_id}, with client: {client_type}")
        if client_type == ClientType.PREFILL:
            return self._determine_prefill_client(request)
        elif client_type == ClientType.DECODE:
            return self._determine_decode_client(request)
        else:
            # Determine the client destination for other stages
            return super()._determine_dst_client(client_type,request)

    def affinity_dermination_low(self, df, num_prefills, num_decodes):
        """
            This function is used to help determine the affinity given the number of prefill/decode clients

            High Affinity means that it determines the parallelism for both prefill and decode and replicate them
            It means that all the interconnects have the same bandwidth
        """
        selected_eng = set()

        # Map 
        low_affinity_client = []
        for i in range (self.num_llm_clients):
            if len(low_affinity_client) == num_prefills:
                break
            row = df[(df['src'] == i) & (df['dst'] < self.num_llm_clients)& (df['dst'] != df['src'])]
            bw_columns = [col for col in df.columns if col.startswith("BW")]
            dst_max_bw = row.loc[row[bw_columns].idxmax()]['dst'].values[0]
            if not dst_max_bw in selected_eng:        
                selected_eng.add(dst_max_bw)
                selected_eng.add(i)
                low_affinity_client.append((i, dst_max_bw))

        # TODO: Need to reorganize the client layouts if this is the case -- 0 is prefill, 1 is decode
        # Move this to the initialization
        for (p, d) in low_affinity_client:
            prefill_client_to_add = LLMClient(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED),
            cluster=self.cluster,
            client_types = [ClientType.PREFILL],
            )
            self.place_client(prefill_client_to_add, [ClientType.PREFILL], p)
            decode_client_to_add = LLMClient(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED),
            cluster=self.cluster,
            client_types = [ClientType.DECODE],
            )
            self.place_client(decode_client_to_add, [ClientType.DECODE], d)

    def affinity_dermination_high(self, num_prefills, num_decodes):
        """
            This function is used to help determine the affinity given the number of prefill/decode clients
            TODO: This is not planned to support for now

            High Affinity means that it determines the parallelism for both prefill and decode and replicate them
            It means that all the interconnects have the same bandwidth
        """ 
        assert num_prefills<=self.num_prefill_clients, f"incorrect prefill clients number"
        assert num_decodes<=self.num_decode_clients, f"incorrect decode clients number"
        num_prefill_reps = math.ceil(self.num_prefill_clients // num_prefills)
        num_decode_reps = math.ceil(self.num_decode_clients // num_decodes)
        assert num_decode_reps == num_decode_reps, f"The number of prefill-decode replications should match"
        logger.debug(f"number of reps: {num_prefill_reps}")
        for i in range(num_prefill_reps):
            strt_p = i*num_prefills
            end_p = min(strt_p + num_prefills, self.num_prefill_clients)
            
            strt_d = i*num_decodes + self.num_prefill_clients
            end_d = min(strt_d + num_decodes, self.num_llm_clients)
            for j_p in range(strt_p, end_p):
                self.prefill_to_decode_matcher[j_p] = []
                for j_d in range(strt_d, end_d):
                    self.prefill_to_decode_matcher[j_p].append(j_d)


    def affinity_client_mapping(self, affinity_groups):
        """
            This function is used to assign the instance clients based on the connection
            
            The input includes the affinity groups of prefill and decode
            format: [([0,1], [4]), ([2,3], [5])] --> the 1st index is prefill and 2nd index is decode
            Then, this function replicate the prefill and decode pairs to limit the 
            accessibility of each prefill-to-decode assignment
        """
        for (p,d) in affinity_groups: 
            for client_id in p:
                self.prefill_to_decode_matcher[client_id] = d

        logger.debug(self.prefill_to_decode_matcher)     
        

            


