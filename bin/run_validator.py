from tensorlink.mpc.nodes import ValidatorNode
import torch.cuda as cuda
import subprocess
import logging
import dotenv
import json
import time
import sys
import os


def get_root_dir():
    if getattr(sys, 'frozen', False):  # Check if running as an executable
        return os.path.dirname(sys.executable)
    else:  # Running as a Python script
        return os.path.dirname(os.path.abspath(__file__))


def check_env_file(_env_path, _config):
    """
    Create a default .env file at the specified path if it doesn't exist.
    """
    if not os.path.exists(_env_path):
        raise ".env does not exist! Create a .env file with PUBLIC_KEY and PRIVATE_KEY as per the documentation."


def load_config(config_path="config.json"):
    try:
        with open(config_path, "r") as f:
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading config: {e}")
        return {}


def main():
    root_dir = get_root_dir()
    env_path = os.path.join(root_dir, ".env")

    # Load config if needed
    config = load_config(os.path.join(root_dir, "config.json"))
    check_env_file(env_path, config)

    local = config.get("local", "false")
    upnp = True
    if local == "true":
        upnp = False

    validator = ValidatorNode(upnp=upnp, local_test=local, off_chain_test=local, print_level=logging.DEBUG)

    try:
        while True:
            time.sleep(5)

            if not validator.node_process.is_alive():
                break

    except KeyboardInterrupt:
        logging.info("Exiting...")


if __name__ == "__main__":
    main()
