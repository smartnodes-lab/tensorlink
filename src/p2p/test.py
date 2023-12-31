from node import Node
import time
import torch
import torch.distributed as dist


# # List of IP addresses for workers
# worker_ips = ["worker1_ip", "worker2_ip", "worker3_ip"]
#
# init_method = "tcp://" + ",".join(worker_ips) + ":port_number"
#
# dist.init_process_group("gloo", init_method=init_method)
#
#
# def discover_devices():
#     num_cuda_devices = torch.cuda.device_count()
#     devices = []
#
#     for i in range(num_cuda_devices):
#         device_properties = torch.cuda.get_device_properties(i)
#         devices.append({
#             "index": i,
#             "name": device_properties.name,
#             "cuda_cores": device_properties.multi_processor_count * 64  # Assuming 64 CUDA cores per SM
#         })
#
#     return devices
#
#
# def assign_modules(modules, devices, latency_threshold):
#     devices.sort(key=lambda x: x["cuda_cores"], reverse=True)
#
#     assignments = []
#     current_device = 0
#
#     for module in modules:
#         pass


ip = "127.0.0.1"
port = 5026

node = Node(
    host=ip,
    port=port,
    debug=True
)

node2 = Node(
    host=ip,
    port=port + 1,
    debug=True
)

node.start()
node2.start()

node.connect_with_node(ip, port + 1)

time.sleep(0.1)

start_time = str(time.time()).encode()
node.send_to_nodes(start_time)

node.stop()
node2.stop()
