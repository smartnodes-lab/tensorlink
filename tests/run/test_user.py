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
        private_key="60c7966e06088af514bb53a3a0f5ea9f918102e401fc46b104b8b28409749b80",
    )

    user.start()

    val_id = user.contract.functions.validatorHashById(1).call()
    user.connect_node(val_id, "127.0.0.1", 5026)

    model = BertModel.from_pretrained("bert-base-uncased")
    user.request_job(model, handle_layers=True)
