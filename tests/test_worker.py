from src.ml.worker import Worker

from transformers import BertModel
import torch
import time


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


ip = "127.0.0.1"
port = 5026


worker1 = Worker(
    host=ip,
    port=port,
    debug=True
)

worker2 = Worker(
    host=ip,
    port=port + 1,
    debug=True
)


worker1.start()
worker2.start()

worker1.connect_with_node("127.0.0.1", port + 1)
time.sleep(1)

worker2.training = True

# print(f"Worker 1: {worker1.all_nodes}")
# print(f"Worker 2: {worker2.all_nodes}")
#
# print("Sending data to Worker 2")
# worker1.send_to_node(worker1.all_nodes[0], b"SALLLAM" * 5_000 + b"SEEEEELIIIIM!")
# time.sleep(1)
#
# worker2.send_to_node(worker2.all_nodes[0], b"Okay.")
# time.sleep(1)


# Worker 1 acts as master node in this scenario
model = BertModel.from_pretrained("bert-base-uncased")
dummy_input = torch.zeros((1, 1), dtype=torch.long)

optimizer = torch.optim.Adam
worker1.distribute_submodules(model)
worker1.model(dummy_input)

# worker1.stop()
# worker2.stop()
