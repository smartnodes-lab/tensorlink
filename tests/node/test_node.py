from tensorlink.mpc.nodes import ValidatorNode
import time


if __name__ == "__main__":
    validator = ValidatorNode(upnp=False, print_level=10)

    while True:
        try:
            time.sleep(1)

        except KeyboardInterrupt:
            break

    validator.cleanup()
