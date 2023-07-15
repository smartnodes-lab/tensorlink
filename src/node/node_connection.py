import base64
import gzip
import threading
import socket
import time


# Represents a connection between two users
class NodeConnection(threading.Thread):
    def __init__(self, origin, host: str, port: int, sock: socket.socket):
        super(ByFrost, self).__init__()

        self.origin = origin
        self.host = host
        self.port = port
        self.sock = sock

        self.terminate_flag = threading.Event()
        self.sock.settimeout(5.0)

    def compress(self, data):
        compressed = data

        try:
            compressed = base64.b64encode(gzip.compress(data))
        except Exception as e:
            # self.origin
            pass

        return compressed

    def send(self, data):
        pass

    def decompress(self, data):
        data = base64.b64decode(data)

        try:
            data = gzip.decompress(data)
        except Exception as e:
            pass

        return data

    def stop(self):
        self.terminate_flag.set()

    def run(self):
        buffer = b""

        while not self.terminate_flag.is_set():
            chunk = b""

            try:
                chunk = self.sock.recv(4096)

            except socket.timeout:
                pass

            except Exception as e:
                self.terminate_flag.set()

            if chunk != b"":
                buffer += chunk
            else:
                break

        if buffer != b"":
            pass

        self.sock.settimeout(None)
        self.sock.close()
