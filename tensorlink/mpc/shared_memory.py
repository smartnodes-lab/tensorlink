from multiprocessing import shared_memory
from copy import deepcopy
import pickle


def get_from_shared_memory(size, name, encoded=False):
    shm = shared_memory.SharedMemory(name=name)
    buffer = shm.buf[:size]

    if not encoded:
        _object = pickle.loads(buffer.tobytes())
    else:
        _object = buffer.tobytes()

    copied_object = deepcopy(_object)
    del buffer
    del _object
    shm.close()
    shm.unlink()
    return copied_object


def store_in_shared_memory(_object, encoded=False):
    if not encoded:
        object_bytes = pickle.dumps(_object)
    else:
        object_bytes = _object

    size = len(object_bytes)
    shm = shared_memory.SharedMemory(create=True, size=size)
    buffer = shm.buf[:size]
    buffer[:] = object_bytes
    del buffer
    shm.close()
    return size, shm.name
