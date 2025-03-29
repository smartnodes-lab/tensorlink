from tensorlink.nodes.user import User
from tensorlink.nodes.validator import Validator
from tensorlink.nodes.worker import Worker

from multiprocessing import Queue
import pytest


@pytest.fixture(scope="module")
def nodes():
    # Initialize nodes
    worker = Worker(Queue(), Queue(), off_chain_test=True, local_test=True, upnp=False)
    user = User(Queue(), Queue(), off_chain_test=True, local_test=True, upnp=False)
    validator = Validator(
        Queue(), Queue(), off_chain_test=True, local_test=True, upnp=False
    )

    # Yield nodes to be used in tests
    yield worker, user, validator

    # Teardown: Stop nodes after all tests complete
    worker.stop()
    user.stop()
    validator.stop()


def test_node_start(nodes):
    # The nodes are started by the fixture, so you can add assertions or interactions here
    worker, user, validator = nodes

    # Start nodes
    worker.start()
    user.start()
    validator.start()

    # Connect worker and user to validator
    worker.connect_node(validator.rsa_key_hash, validator.host, validator.port)
    user.connect_node(validator.rsa_key_hash, validator.host, validator.port)

    assert (
        worker.is_alive()
    )  # Example: you should have a method to check the node state
    assert user.is_alive()
    assert validator.is_alive()


# def test_job_request(nodes):
#     worker, user, validator = nodes
#
#     user.request_job(1, 1, {
#         "type": "offloaded",
#         "id_hash": "",
#         "module": f"dummy".split(".")[-1].split(">")[0][
#             :-1
#         ],  # class name
#         "mod_id": -1,
#         "size": 1e8,
#         "workers": [],
#         "training": True,
#     })
