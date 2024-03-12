from transformers import BertModel
import torch.nn as nn
import torch


# model = BertModel.from_pretrained("bert-base-uncased")

model = nn.Sequential(
    nn.Linear(10, 100),
    nn.Linear(100, 100),
    nn.Linear(100, 10)
)

op1 = torch.optim.Adam(layer.parameters())
op2 = torch.optim.Adam(layer2.parameters())

input_vec = torch.zeros((1, 10)).clone().detach()

output1 = layer1(input_vec)
output1.retain_grad()

intermediate1 = output1.clone().detach().requires_grad_()

output2 = layer2(intermediate1)
output2.retain_grad()

pre_updated = layer2.weight

loss2 = output2.sum()
loss2.backward()
op2.step()
op2.zero_grad()

post_updated = layer2.weight

print(torch.allclose(pre_updated, pre_updated))

output1.backward(intermediate1.grad)
op1.zero_grad()
op1.step()
