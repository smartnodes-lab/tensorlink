from transformers import BertModel
from src.roles.user import User


if __name__ == "__main__":
    ip = "127.0.0.1"

    user = User(
        ip,
        5029,
        debug=True,
        upnp=False,
        off_chain_test=False,
        private_key="119827f707499580e9bd124a67610f0e0d223d4731f4e73c1d6bec5b1841a48b",
    )

    user.start()

    val_id = user.contract.functions.validatorKeyById(1).call()
    user.connect_node(val_id, "127.0.0.1", 5026)

    model = BertModel.from_pretrained("bert-base-uncased")
    user.request_job(model, handle_layers=True)
