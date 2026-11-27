from mist.Tracing import ChromeTracingLogger
from mist.Client import MISTClient,ClientType, stage_to_client_mapping,ClientMetrics
from mist.Clusters.clusters import ClusterConfig, ClusterType
from mist.Scheduler.scheduler import Scheduler, SchedulerConfig, BatchingMethod
from mist.Request import Request, DataMetrics, RequestMetrics, RequestStatus, RequestStage
from mist.Input_requests.Request_inputs import RequestDistributions, UniformDistribution, PoissonDistribution, NormalDistribution
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
import heapq
from itertools import chain
from uuid import uuid4
import pandas as pd
import numpy as np
import enum
import math

from mist.Coordinator.global_router import CoordRouterType, LoadTypes, req_is_heavy
from mist.Global_Network.astrasim_network import AstraSimNetwork, NS_PER_MS
import logging

logger = logging.getLogger(__name__)


class EventType(enum.Enum):
    """Status of a sequence."""
    # At equal times, events run in this order: a transfer that finishes at T is
    # handed to its client before that client starts a batch at T.
    STAGE_PUSH = enum.auto()
    NETWORK_STEP = enum.auto()
    NETWORK_SEND = enum.auto()
    CLIENT_STEP = enum.auto()

    def __lt__(self, other):
        return self.value < other.value

class MISTCoordinator:
    def __init__(
            self,
            starting_request_queue: Optional[List[Request]] = None,
            logging_file: Optional[str] = 'trace.json',
            max_sim_time: Optional[float] = np.inf,
            network_file: Optional[str | pd.DataFrame] = None,
            router_type: Optional[CoordRouterType] = CoordRouterType.ROUND_ROBIN,
            astrasim_topology: Optional[str] = None,
        ):

        self.router_type = router_type

        ## Client related parameters
        self.num_total_clients = 0
        self.clients= []
        self.client_matcher = dict()
        self.client_active = []
        self.client_next_step = dict()
        self.client_connection_bw_latency_df = pd.DataFrame()
        self.network_file = network_file

        ## ASTRA-sim network backend: when set, transfers between different clients
        ## go through it (with congestion) instead of the analytical link model.
        self.network = AstraSimNetwork(astrasim_topology) if astrasim_topology else None
        self._in_flight = dict()        # request_id -> (req, src client, dst client, send time ms, record KV time)
        self._network_steps = set()     # MIST times with a NETWORK_STEP already queued

        ## Request related parameters
        self.starting_request_queue = [] if starting_request_queue is None else starting_request_queue
        self.request_accepted = len(self.starting_request_queue)
        self.event_queue = []
        self.request_serviced = 0
        self.completed_requests = []
        self._active_requests: dict[int, Request] = {}

        ## Simulation related parameters
        self.client_sim_queue = []
        self.global_time = 0
        self.max_sim_time = np.inf if max_sim_time is None else max_sim_time
        self._last_event_time = 0

        ## Logger
        self.logger = ChromeTracingLogger(logging_file)

    def add_client(self, client:MISTClient, client_types:List[ClientType]):
        """
            This function is designed to add an client to the MIST system
        """
        client_id = self.num_total_clients
        client.client_id = client_id
        self.clients.append(client)
        client.logger = self.logger
        for client_type in client_types:
            if client_type not in self.client_matcher:
                self.client_matcher[client_type] = []
            self.client_matcher[client_type].append(client_id)
        self.num_total_clients += 1
        self.client_active.append(False)
        self.client_next_step[client_id] = 0
        # heapq.heappush(self.client_sim_queue, (0, client_id))

    def add_request(self, request:Request, time:float):
        """
            This function is designed to add a request to the MIST system
        """
        request.request_id = self.request_accepted
        self.request_accepted += 1
        heapq.heappush(self.request_queue, (time, request, self._get_current_client(request)))

    def _get_current_client(self, request: Request):
        """
            This function is designed to get the current client based on the request
        """
        current_stage = request.current_stage
        client_type_required = stage_to_client_mapping(current_stage)
        client_id = self._determine_dst_client(client_type_required, request)
        return client_id

    def _get_next_client(self, request: Request):
        """
            This function is designed to get the next client based on the request
        """
        next_stage = request.get_next_stage()
        client_type_required = stage_to_client_mapping(next_stage)
        client_id = self._determine_dst_client(client_type_required, request)
        # print(f"Request {request.request_id} current stage {request.current_stage} next stage {next_stage}, client type {client_type_required}, client {client_id}")
        return client_id

    def run_sim(self):
        """Process events before the cutoff and count work from steps already started.

        Client steps run synchronously. A step admitted before `max_sim_time`
        can finish after it, without admitting the next event.
        """
        ## Initialize the DF for client connection
        # self.initial_client_connection_df("network_bw.csv")
        if isinstance(self.network_file, pd.DataFrame):
            self.client_connection_bw_latency_df = self.network_file
        elif isinstance(self.network_file, str):
            self.initial_client_connection_from_file()
        else:
            self.initial_client_connection_df()
        if self.network is not None and self.network.npus_count < self.num_total_clients:
            raise ValueError(f"ASTRA-sim topology has {self.network.npus_count} NPUs but the coordinator "
                             f"has {self.num_total_clients} clients; client i maps to NPU i.")

        next_print_time = 1000

        ## Client initialization might change the first client for each request
        ## Determine the first client for each request
        for req_id in range(self.request_accepted):
            self.starting_request_queue[req_id].request_id = req_id
            heapq.heappush(self.event_queue, (self.starting_request_queue[req_id].metrics.arrival_time,
                                            EventType.STAGE_PUSH, (self.starting_request_queue[req_id], None)))

        queued_client_events = set()
        while (
            self.event_queue
            and self.request_serviced < self.request_accepted
            and self.event_queue[0][0] < self.max_sim_time
        ):
            ## Add all the requests to appropriate clients
            # Pop out the event and determine the type
            event_push_time, event_type, event_items = heapq.heappop(self.event_queue)
            self._last_event_time = event_push_time
            # print(f"Global Time {self.global_time}, Next Event Start Time: {event_push_time}, Event Type: {event_type}")

            if (event_type == EventType.STAGE_PUSH):
                (cur_req, next_client_id) = event_items
                if next_client_id == None:
                    ## First client should be determined only once the req has arrived
                    next_client_id = self._get_current_client(cur_req)
                self._active_requests[cur_req.request_id] = cur_req

                request_added_client_time = self.clients[next_client_id].current_time
                if request_added_client_time < event_push_time:
                    ##
                    request_added_client_time = event_push_time
                # push the run client event into the queue
                req_push_event = (request_added_client_time, next_client_id)
                ## If the client is active but will be active later, update the sim queue
                if (self.client_next_step[next_client_id] > request_added_client_time) or self.clients[next_client_id].client_idle():
                    self.client_next_step[next_client_id] = request_added_client_time
                    # if next_client_id == 1:
                    #     print(f"Request added to request queue {next_client_id} at {request_added_client_time}")
                    heapq.heappush(self.event_queue, (request_added_client_time, EventType.CLIENT_STEP, req_push_event))
                    queued_client_events.add(req_push_event)

                # TODO: Here it uses the event push time but it might have delay between arrival time
                # and the actual added request time
                self.clients[next_client_id].add_request(cur_req, event_push_time)

            elif (event_type == EventType.CLIENT_STEP):
                (cur_time,client_id) = event_items
                if event_items in queued_client_events:
                    queued_client_events.remove(event_items)
                self.global_time = cur_time
                # print(f"Client {client_id} is idle =  {self.clients[client_id].client_idle()}")
                # if self.clients[client_id].client_idle() == False:
                if self.clients[client_id].client_idle() == False:
                    # if client_id == 1:
                    #     print(f"Normal Client {client_id} step to start at {cur_time}")
                    client_step_end_time, req_list = self.clients[client_id].step(cur_time)
                    self.global_time = cur_time
                else:
                    continue
                # if self.request_queue:
                #     self.global_time = min(self.global_time, self.request_queue[0][0])
                if self.global_time >= next_print_time:
                    logger.debug(f"Global Time {self.global_time}")
                    next_print_time = (self.global_time // 1000 + 1) * 1000
                eng_push_event = (client_step_end_time, client_id)
                if self.clients[client_id].client_idle() == False:
                    # if client_id == 1:
                    #     print(f"Normal Client {client_id} step end at {client_step_end_time}")
                    heapq.heappush(self.event_queue, (client_step_end_time, EventType.CLIENT_STEP, eng_push_event))
                    self.client_next_step[client_id] = client_step_end_time

                # Handling the finished requests list and reinsert them into the event queue as needed
                if req_list:
                    for req in req_list:
                        request_client_exit_time = req.get_client_exit_time()
                        # print(f'Req {req.request_id} given out by client {client_id} at step end. {req.request_finished()}')
                        if req.request_finished():
                            self.request_serviced += 1
                            req.update_request_finished_time(request_client_exit_time)
                            self._active_requests.pop(req.request_id, None)
                            self.completed_requests.append(req)
                            # print(f'Req {req.request_id} serviced')
                        else:
                            next_client_id = self._get_next_client(req)
                            if self.network is not None and client_id != next_client_id:
                                self._queue_network_send(req, client_id, next_client_id, request_client_exit_time)
                                continue
                            self.logger.log_event(req.request_id, 'Coordinator', str(client_id)+"->"+str(next_client_id), time=request_client_exit_time,type= "B")
                            movement_time = self._get_move_request_time(req, client_id, next_client_id)
                            self.logger.log_event(req.request_id, 'Coordinator', str(client_id)+"->"+str(next_client_id), time=request_client_exit_time+movement_time,type= "E")
                            # print(f'Req {req.request_id} pushed to {next_client_id} at {request_client_exit_time+movement_time}')
                            req.current_stage = req.get_next_stage()
                            # Handle requests that are needed to be reinserted to the client
                            req_push_event = (req, next_client_id)
                            heapq.heappush(self.event_queue, (request_client_exit_time+movement_time, EventType.STAGE_PUSH, req_push_event))
            elif (event_type == EventType.NETWORK_SEND):
                self._network_send(event_push_time, *event_items)
            elif (event_type == EventType.NETWORK_STEP):
                self._network_steps.discard(event_push_time)
                self._advance_network(event_push_time)
            else:
                raise ValueError(f"No event type found for the client:{event_type}")
        if self.logger.filename is not None:
            self.logger.save()
        ## Log the completion of the simulation
        self.global_stats = self.get_global_stats()

    ## ASTRA-sim transfers. ASTRA-sim's clock (ns) is kept equal to MIST's clock (ms).
    def _queue_network_send(self, req, src_client, dst_client, exit_time):
        """Decide the transfer now; start it in ASTRA-sim when MIST time reaches `exit_time`."""
        size_bytes = max(1, math.ceil(req.get_data_movement_size(self.clients[dst_client].model)))
        record_kv = req.current_stage == RequestStage.PREFILL
        req.current_stage = req.get_next_stage()
        heapq.heappush(self.event_queue, (exit_time, EventType.NETWORK_SEND,
                                          (req, src_client, dst_client, size_bytes, record_kv)))

    def _network_send(self, now, req, src_client, dst_client, size_bytes, record_kv):
        self._advance_network(now)
        self.logger.log_event(req.request_id, 'Coordinator', f"{src_client}->{dst_client}", time=now, type="B")
        self._in_flight[req.request_id] = (req, src_client, dst_client, now, record_kv)
        self.network.send(now, size_bytes, src_client, dst_client, req.request_id)
        self._schedule_network_step()

    def _advance_network(self, now):
        """Run ASTRA-sim up to `now` and hand finished transfers to their destination client."""
        for t, request_id in self.network.advance(now):
            req, src_client, dst_client, sent_ms, record_kv = self._in_flight.pop(request_id)
            if record_kv:
                req.metrics.kv_transfer_time = t - sent_ms
            self.logger.log_event(req.request_id, 'Coordinator', f"{src_client}->{dst_client}", time=t, type="E")
            heapq.heappush(self.event_queue, (t, EventType.STAGE_PUSH, (req, dst_client)))
        self._schedule_network_step()

    def _schedule_network_step(self):
        t = self.network.next_event_ms()
        if t is not None and t not in self._network_steps:
            self._network_steps.add(t)
            heapq.heappush(self.event_queue, (t, EventType.NETWORK_STEP, None))

    def get_global_stats(self) -> ClientMetrics:
        total_input = 0
        output_tokens = 0
        time_to_first_token = 0
        requests_with_tokens = 0
        decode_time = 0
        decode_steps = 0
        latest_client_time = max((client.current_time for client in self.clients), default=0)
        latest_finished_time = max(self.global_time, self._last_event_time, latest_client_time)

        # Admitted requests stay visible while they wait for a chunk or transfer.
        for request in chain(self.completed_requests, self._active_requests.values()):
            total_input += request.input_len - request.remaining_prefill_tokens
            output_tokens += request.gen_tokens
            if request.data:
                requests_with_tokens += 1
                time_to_first_token += request.data[0].finished_time - request.metrics.arrival_time
                for index in range(1, len(request.data)):
                    beat = request.data[index]
                    decode_time += beat.finished_time - beat.scheduled_time
                decode_steps += len(request.data) - 1

        e2e_latencies = []
        for request in self.completed_requests:
            e2e_latencies.append(request.metrics.finished_time - request.metrics.arrival_time)
            latest_finished_time = max(latest_finished_time, request.metrics.finished_time)

        avg_TTFT = time_to_first_token / requests_with_tokens if requests_with_tokens else 0
        avg_TPOT = decode_time / decode_steps if decode_steps else 0
        per_second = 1000 / latest_finished_time if latest_finished_time > 0 else 0
        num_requests = len(e2e_latencies)
        if num_requests:
            e2e_latencies.sort()
            T50_latency = e2e_latencies[int(0.5 * num_requests)]
            T90_latency = e2e_latencies[int(0.9 * num_requests)]
            T95_latency = e2e_latencies[int(0.95 * num_requests)]
            T99_latency = e2e_latencies[int(0.99 * num_requests)]
        else:
            T50_latency = T90_latency = T95_latency = T99_latency = 0

        return ClientMetrics(
            TTFT=avg_TTFT,
            TPOT=avg_TPOT,
            rps=num_requests * per_second,
            T50_latency=T50_latency,
            T90_latency=T90_latency,
            T95_latency=T95_latency,
            T99_latency=T99_latency,
            interactivity=1000 / avg_TPOT if avg_TPOT > 0 else 0,
            output_throughput=output_tokens * per_second,
            total_token_throughput=(total_input + output_tokens) * per_second,
        )

    def get_heavy_light_split(self, client_type, req, load_type):
        def circular_left_shift(lst, indexes):
            """Perform a circular left shift on the specified indexes of the list."""
            if not indexes:
                return lst
            first = lst[indexes[0]]
            for i in range(len(indexes) - 1):
                lst[indexes[i]] = lst[indexes[i + 1]]
            lst[indexes[-1]] = first
            return lst

        if req_is_heavy(req, load_type):
            # Circular left shift on even indexes
            client_id = self.client_matcher[client_type][0]
            even_indexes = [i for i in range(len(self.client_matcher[client_type])) if i % 2 == 0]
            self.client_matcher[client_type] = circular_left_shift(self.client_matcher[client_type], even_indexes)
        else:
            # Circular left shift on odd indexes
            client_id = self.client_matcher[client_type][1]
            odd_indexes = [i for i in range(len(self.client_matcher[client_type])) if i % 2 != 0]
            self.client_matcher[client_type] = circular_left_shift(self.client_matcher[client_type], odd_indexes)
        return client_id

    def get_load_balanced_client(self, client_type, load_type):
        min_tokens = np.inf
        client_id = -1
        for client in self.client_matcher[client_type]:
            client_load = self.clients[client].get_load(self.global_time, load_type)
            if  client_load < min_tokens:
                min_tokens = client_load
                client_id = client
        return client_id

    ## Overloadable functions
    def _determine_dst_client(self, client_type:ClientType, request=-1):
        """
            This function is designed to determine the destination client based on the client type.
            By default, it uses a round robin policy.
            This function can be overloaded to use different coordinator policies
        """
        if client_type in self.client_matcher:
            if self.router_type == CoordRouterType.ROUND_ROBIN:
                ## Round robin policy
                # Find the least recently used client_id
                ## If the request has a preference, use that
                if isinstance(request, Request) and  request.get_client_preference():
                    client_ids_requested = request.get_client_preference()
                    client_id = next((client for client in self.client_matcher[client_type] if client in client_ids_requested), None)
                    if client_id is not None:
                        self.client_matcher[client_type].remove(client_id)
                    if client_id is None:
                        raise ValueError(f"No matching client found for the requested client IDs: {client_ids_requested}")
                ## Else use the round robin policy
                else:
                    client_id = self.client_matcher[client_type].pop(0)
                # Rotate the list by 1
                self.client_matcher[client_type].append(client_id)
            # We assume that the total number of output tokens is known beforehand for the request and we balance the sum of decode tokens on both the model instances.
            elif self.router_type == CoordRouterType.DECODE_BALANCER:
                ## Find the client with the least number of decode tokens
                client_id = self.get_load_balanced_client( client_type, LoadTypes.DECODE_TOKENS)
            # Dedicated half instance for servicing only the heavy-decode requests and other half services only the light-decode requests.
            elif self.router_type == CoordRouterType.DECODE_HEAVY_LIGHT:
                client_id = self.get_heavy_light_split( client_type, request, LoadTypes.DECODE_TOKENS)
            # Keep number of prefill tokens balanced across all the model instances
            elif self.router_type == CoordRouterType.PREFILL_BALANCER:
                ## Find the client with the least number of prefill tokens
                client_id = self.get_load_balanced_client(client_type, LoadTypes.PREFILL_TOKENS)
            # Dedicated instance for heavy prefills and the other for light prefills
            elif self.router_type == CoordRouterType.PREFILL_HEAVY_LIGHT:
                client_id = self.get_heavy_light_split(client_type, request, LoadTypes.PREFILL_TOKENS)
            # Keep load balanced across all the model instances
            elif self.router_type == CoordRouterType.CHUNKED_LOAD_BALANCER:
                client_id = self.get_load_balanced_client(client_type, LoadTypes.TOKEN_PROCESSING)
            # Dedicated half machines for heavy loads and other half for light loads
            elif self.router_type == CoordRouterType.CHUNKED_HEAVY_LIGHT:
                client_id = self.get_heavy_light_split(client_type, request, LoadTypes.TOKEN_PROCESSING)
            elif self.router_type == CoordRouterType.ORACLE_BALANCER:
                client_id = self.get_load_balanced_client(client_type, LoadTypes.ORACLE_DECODE)
            elif self.router_type == CoordRouterType.RUNTIME_LOAD_BALANCER:
                client_id = self.get_load_balanced_client(client_type, LoadTypes.SIM_TIME)
            else:
                raise ValueError(f"Invalid router type {self.router}")
            # print(f"Request {request.request_id}, {request.input_len}/{request.output_len} for {client_type}: Sent to client {client_id}")
            return client_id
        else:
            raise ValueError(f"No client type found for the stage:{client_type}")

    def _update_most_recent_client(self, client_type, client_id):
        """
            The input includes the client type and the client id that was most recently used.
        """
        assert client_id != -1 and client_id is not None, "Invalid client id to update the most recent used"
        # Find the least recently use
        self.client_matcher[client_type].remove(client_id)
        # Rotate the list by 1
        self.client_matcher[client_type].append(client_id)

    def compute_bw_latency(self, df: pd.DataFrame, src, dst):
        # Filter the row corresponding to the given src and dst
        row = df[(df['src'] == src) & (df['dst'] == dst)]

        if row.empty:
            raise ValueError(f"No connection found from src:{src} to dst:{dst}")

        ## Assuming no congestion, we can use the minimum bandwidth and total latency
        # Extract bandwidth and latency columns
        bw_columns = [col for col in df.columns if col.startswith("BW")]
        latency_columns = [col for col in df.columns if col.startswith("latency")]

        # Compute minimum bandwidth and total latency
        min_bw = row[bw_columns].min(axis=1).values[0]
        total_latency = row[latency_columns].sum(axis=1).values[0]

        return total_latency, min_bw

    # TODO: Use this as a movement between client
    def _get_move_request_time(self, request, src_client, dst_client) -> float:
        # Get the cost of moving the request from src to dst client
        data_size = request.get_data_movement_size(self.clients[dst_client].model)/2**20   # Convert to MB
        if src_client == dst_client:
            # Update the KV Cache Transfer Time
            if request.current_stage == RequestStage.PREFILL:
                request.metrics.kv_transfer_time = 0
            
            return 0
        else:
            latency, BW = self.compute_bw_latency(self.client_connection_bw_latency_df, src_client, dst_client)
            # print(f'Req:{request.request_id}, Src: {src_client}, Dst: {dst_client}, Data Size: {data_size}, Latency: {latency}, BW: {BW}')
            if request.current_stage == RequestStage.PREFILL:
                request.metrics.kv_transfer_time = latency + data_size/BW
            return latency + data_size/BW

    ## Overloadable functions
    def initial_client_connection_df(self):
        # Initialize the client connection dataframe
        ## All BWs should be in GB/s and latencies in msec
        data = []
        for src_client in range(self.num_total_clients):
            for dst_client in range(self.num_total_clients):
                if src_client == dst_client:
                    data.append([src_client, dst_client, 0,  0])
                else:
                    data.append([src_client, dst_client, 5, 10])
        self.client_connection_bw_latency_df = pd.DataFrame(data, columns=['src', 'dst', 'latency(msec)', 'BW(GB/s)'])

    def initial_client_connection_from_file(self):
        file_name = self.network_file
        if isinstance(file_name, str) and file_name.endswith('.csv'):
            with open(file_name, 'r') as f:
                network_fields = f.readline().strip()  # First line has names (may have multiple fields for latency)
                column_names = network_fields.split(',')
            self.client_connection_bw_latency_df = pd.read_csv(file_name, skiprows=1, names=column_names)
        else:
            raise ValueError(f"Invalid file name {file_name}")


    def _add_request_to_client(self, request, client_id):
        # The arrival time is based on priority time
        self.clients[client_id].add_request(request, request.priority_time)
        None




