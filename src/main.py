from substrateinterface import SubstrateInterface
import os


# Read seed from file in dir
def get_local_seed():
    if not os.path.exists("seed.txt"):
        raise FileNotFoundError("seed.txt not found!")

    with open("seed.txt", "r") as file:
        return file.read().strip()


if __name__ == "__main__":
    # Connect to the ml-net contract and await jobs?
    url = "wss://ws.test.azero.dev"
    seed = get_local_seed()
    substrate = SubstrateInterface(url=url)
