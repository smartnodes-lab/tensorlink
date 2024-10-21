from tensorlink.mpc.nodes import ValidatorNode
import time


if __name__ == "__main__":
    validator = ValidatorNode()

    while True:
        try:
            time.sleep(1)

        except KeyboardInterrupt:
            break

    validator.cleanup()
