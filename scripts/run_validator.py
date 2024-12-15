from tensorlink.mpc.nodes import ValidatorNode
import logging
import time


if __name__ == "__main__":
    validator = ValidatorNode(upnp=True, print_level=logging.INFO)

    while True:
        try:
            time.sleep(1)

        except KeyboardInterrupt:
            break

    validator.cleanup()
