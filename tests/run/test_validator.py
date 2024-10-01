import time

from src.mpc.coordinator import ValidatorCoordinator


if __name__ == "__main__":

    validator = ValidatorCoordinator(
        debug=True,
        upnp=True,
        off_chain_test=True
    )

    while True:

        try:
            time.sleep(3)

        except KeyboardInterrupt:
            break

    validator.stop()
