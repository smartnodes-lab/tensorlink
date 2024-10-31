from tensorlink.mpc.nodes import WorkerNode, ValidatorNode, UserNode
from tensorlink.crypto.rsa import *
import hashlib
from transformers import BertModel
import time


if __name__ == "__main__":

    # user = UserNode(debug=True, upnp=False, off_chain_test=True)
    # time.sleep(0.2)
    # worker = WorkerNode(debug=True, upnp=False, off_chain_test=True)
    # time.sleep(0.2)
    validator = ValidatorNode(debug=True, upnp=False, off_chain_test=True)

    time.sleep(0.2)

    # val_key, val_host, val_port = validator.send_request("info", None)

    # worker.send_request("connect_node", (val_key, val_host, val_port))
    # time.sleep(5)
    # user.send_request("connect_node", (val_key, val_host, val_port))
    # user.send_request("connect_node", (b"", "142.188.24.158", 38751))

    # model = BertModel.from_pretrained("bert-base-uncased")
    # distributed_model = user.create_distributed_model(model, 1, 1)
    # for _ in range(10):
    #     din = torch.zeros((1, 32), dtype=torch.long)
    #     output = distributed_model(din)
    # loss = output.last_hidden_state
    # distributed_model.backward(loss)

    validator.cleanup()
    # worker.cleanup()
    # user.cleanup()
