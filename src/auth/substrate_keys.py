from substrateinterface import Keypair
import base64
import json


def load_substrate_keypair(public_key, password):
    try:
        with open(f"keys/{public_key}.json", "r") as f:
            exported_account = json.load(f)

        keypair = Keypair.create_from_encrypted_json(exported_account, password)

    except:
        print("Could not find substrate keys, utilizing test account: Alice")
        keypair = Keypair.create_from_uri("//Alice")

    return keypair
