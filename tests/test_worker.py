from src.ml.worker import Worker

from transformers import BertModel
from src.ml.model_analyzer import handle_output
import torch.nn as nn
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


master = Worker(
    host=ip,
    port=port,
    debug=True
)


worker1 = Worker(
    host=ip,
    port=port + 1,
    debug=True
)


master.start()
worker1.start()

master.connect_with_node("127.0.0.1", port + 1)
worker1.connect_with_node("127.0.0.1", port)
time.sleep(1)

master.training = True
worker1.training = True

# print(f"Worker 1: {worker1.all_nodes}")
# print(f"Worker 2: {worker2.all_nodes}")
#
# print("Sending data to Worker 2")
# worker1.send_to_node(worker1.all_nodes[0], b"SALLLAM" * 5_000 + b"shallom!")
# time.sleep(1)
#
# worker2.send_to_node(worker2.all_nodes[0], b"Okay.")
# time.sleep(1)


def custom_backward(s):
    print("POOP")


# Worker 1 acts as master node in this scenario
model = BertModel.from_pretrained("bert-base-uncased")
model.backward = custom_backward

# model = nn.Sequential(nn.Linear(10, 6000000), nn.Linear(6000000, 2))
dummy_input = torch.zeros((1, 1), dtype=torch.long)

optimizer = torch.optim.Adam
master.distribute_model(model)
out = master.model.forward(dummy_input)
print(out)
# out.sum().backward()


# layer1 = nn.Linear(10, 100)
# layer2 = nn.Linear(100, 10)
# inp = torch.zeros((1, 10))
#
# op1 = torch.optim.Adam(layer1.parameters())
# op2 = torch.optim.Adam(layer2.parameters())
#
# out1 = layer1(inp)
# intermediate = out1.clone().detach().requires_grad_()
#
# out2 = layer2(intermediate)
# out2.retain_grad()
# loss2 = out2.sum()
# loss2.backward()
#
# out1.backward(intermediate.grad, retain_graph=True)
#
# op1.zero_grad()
# op2.zero_grad()
# op1.step()
# op2.step()

master.stop()
worker1.stop()
