from substrateinterface import SubstrateInterface
import os


# Read seed from file in dir
def get_local_seed():
    if not os.path.exists("SEED.txt"):
        raise FileNotFoundError("SEED.txt not found!")

    with open("SEED.txt", "r") as file:
        return file.read().strip()


if __name__ == "__main__":
    url = "wss://ws.test.azero.dev"
    seed = get_local_seed()
    substrate = SubstrateInterface(url=url)
