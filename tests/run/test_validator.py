from src.roles.validator import Validator


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 5026

    validator = Validator(
        ip,
        port,
        debug=True,
        upnp=False,
        off_chain_test=False,
        private_key="892938adae985cde8831026ceb6d312d9c81dfe7c1d4aba8d7bd3f219bd81385",
    )

    validator.start()
