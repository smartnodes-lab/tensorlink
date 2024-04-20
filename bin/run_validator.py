from src.roles.validator import Validator
from src.roles.user import User

from transformers import BertModel


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 5026

    validator = Validator(
        ip, port, "5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh", True
    )

    user = User(
        host=ip,
        port=5025,
        wallet_address="5HDxH5ntpmr7U3RjEz5g84Rikr93kmtqUWKQum3p3Kdot4Qh",
        debug=True,
    )

    validator.start()
    user.start()

    user.connect_dht_node(ip, port)

    user.bootstrap()

    user.stop()
    validator.stop()

    # model = BertModel.from_pretrained("bert-base-uncased")
    # d_model = user.request_job(model, 1.4e9)
