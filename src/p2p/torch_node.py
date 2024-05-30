from src.ml.model_analyzer import get_gpu_memory
from src.p2p.smart_node import SmartNode
from src.p2p.connection import Connection

import pickle


class TorchNode(SmartNode):
    def __init__(
        self,
        host: str,
        port: int,
        debug: bool = False,
        max_connections: int = 0,
        upnp=True,
        off_chain_test=False,
    ):
        super(TorchNode, self).__init__(
            host,
            port,
            debug=debug,
            max_connections=max_connections,
            upnp=upnp,
            off_chain_test=off_chain_test,
        )

        # Available GPU memory estimation
        self.available_memory = get_gpu_memory()

        # Model parameters
        self.modules = {}
        self.optimizers = {}
        self.parameters = {}
        self.state_updates = {}

        # Master flag for handling different types of storage as master
        self.master = False

    def handle_data(self, data: bytes, node: Connection):
        try:
            handled = super().handle_data(data, node)
            ghost = 0

            if not handled:
                if b"LOADED" == data[:6]:
                    pickled = data[6:]
                    self.debug_print(
                        f"Successfully offloaded submodule to: {node.node_id}"
                    )
                elif b"FORWARD" == data[:7]:
                    # Received a forward pass
                    self.debug_print(
                        f"RECEIVED FORWARD: {round((data.__sizeof__() - 5) / 1e6), 1} MB"
                    )

                    if self.master:
                        # TODO we must check that the forward received corresponds to a sent pass/specific module
                        [n_iter, n_micro, module_id], tensor = pickle.loads(data[7:])
                        self.modules["Master"].forward_queues[n_micro].put(
                            ([n_iter, n_micro, module_id], tensor)
                        )

                    # TODO we must check that the forward received corresponds to a sent pass/specific module
                    elif self.modules:
                        (n_iter, n_micro, module_id), tensor = pickle.loads(data[7:])
                        self.modules[module_id].forward_queues.put(
                            ([n_iter, n_micro], tensor)
                        )

                elif b"BACKWARD" == data[:8]:
                    # TODO same with backwards pass
                    self.debug_print(
                        f"RECEIVED BACKWARD: {round((data.__sizeof__() - 5) / 1e6, 1)} MB"
                    )

                    # Master-specific handling (ie for DistributedModel)
                    if self.master:
                        [n_iter, n_micro, module_id], tensor = pickle.loads(data[8:])
                        self.modules["Master"].backward_queues[n_micro].put(
                            ([n_iter, n_micro, module_id], tensor)
                        )

                    # Module-specific handling (ie for OffloadedModule / nn.Module)
                    elif self.modules:
                        (n_iter, n_micro, module_id), tensor = pickle.loads(data[8:])
                        self.modules[module_id].backward_queues.put(
                            ([n_iter, n_micro], tensor)
                        )

                # Handle requests for module parameters
                elif b"PARAMS-REQ" == data[:10]:
                    self.debug_print(f"RECEIVED PARAMS REQUEST")

                    # TODO Must ensure requesting node is indeed the master or an overseeing validator
                    module_id = data[10:]
                    self.send_parameters(
                        node, self.modules[module_id].parameters(), module_id
                    )

                    return True

                # Handle and store responses from a parameters request
                elif b"PARAMETERS" == data[:10]:
                    self.debug_print(f"RECEIVED PARAMS REQUEST")
                    module_id, parameters = pickle.loads(data[10:])
                    self.parameters[module_id] = parameters

                else:
                    # We do not log a ghost here since SmartNode is meant to be a super class and this should
                    # only be invoked by a super call
                    return False

            if ghost > 0:
                self.update_node_stats(node.node_id, "GHOST")
                # TODO: potentially some form of reporting mechanism via ip and port

            return True

        except Exception as e:
            self.debug_print(f"handle_data: Error handling data: {e}")

    def send_forward(self, node: Connection, args, context):
        """Send forward pass to node, must contain args (module args) and context (module + epoch id)"""
        pickled_data = b"FORWARD" + pickle.dumps((context, args))
        self.send_to_node(node, pickled_data)

    def send_backward(self, node: Connection, args, context):
        """Send backward pass to node, must contain args (module args) and context (module + epoch id)"""
        pickled_data = b"BACKWARD" + pickle.dumps((context, args))
        self.send_to_node(node, pickled_data)

    def send_parameters(self, node: Connection, parameters, module_id):
        """Send specific module parameters
        TODO should be accompanied by a requested proof (from smart contract) or the specific user
        """
        pickled_data = b"PARAMETERS" + pickle.dumps((module_id, list(parameters)))
        self.send_to_node(node, pickled_data)

    def send_parameters_req(self, node: Connection, module_id):
        """Request parameters from a specific worker"""
        self.send_to_node(node, b"PARAMS-REQ" + module_id)
