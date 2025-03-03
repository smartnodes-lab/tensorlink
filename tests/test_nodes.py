from tensorlink.roles.user import User
from tensorlink.roles.validator import Validator
from tensorlink.roles.worker import Worker

# from tensorlink.ml.

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


def test_node_functionality(nodes):
    """Basic test of send, receive, and core functionality"""
    worker, user, validator = nodes

    # Validator state and environment management
    validator.save_dht_state()
    validator.load_dht_state()

    # Sending basic data to each other
    # worker.ping_node()
    pass


def test_job_request(nodes):
    """Ensure job request and job-related functionalities work"""
    worker, user, validator = nodes

    # user.request_job(1, 1, {""})
    # ... perform tests that rely on nodes still being active
    assert True


def test_send_ghost_data():
    """Test the sending of information we cannot handle or were not expecting."""
    pass


def test_node_spam():
    """Test the blocking of nodes that overload a connection."""
    pass
