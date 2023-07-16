import base64
import gzip
import threading
import socket
import time


# Represents a connection between two nodes
class NodeConnection(threading.Thread):
    def __init__(self, connection, sock: socket.socket, id: str, host: str, port: int):
        super(NodeConnection, self).__init__()

        self.connection = connection
        self.id = id
        self.host = host
        self.port = port
        self.sock = sock

        self.EOT_CHAR = 0x04.to_bytes(1, 'big')
        self.COMPRESS_CHAR = 0x02.to_bytes(1, 'big')

        self.terminate_flag = threading.Event()
        self.sock.settimeout(5.0)

        self.connection.debug_print(f"node-connection::starting::{self.id}::{self.host}:{self.port}")

    def compress(self, data):
        self.connection.debug_print(f"node-connection::{self.id}::compress::input::{data}")
        compressed = data

        try:
            compressed = base64.b64encode(gzip.compress(data))
            self.connection.debug_print(f"node-connection::{self.id}::compress::done")
        except Exception as e:
            self.connection.debug_print(f"node-connection::{self.id}::compress::error: {e}")

        return compressed

    def decompress(self, data):
        self.connection.debug_print(f"node-connection::{self.id}::decompress::decode")
        data = base64.b64decode(data)

        try:
            self.connection.debug_print(f"node-connection::{self.id}::decompress::data")
            data = gzip.decompress(data)
        except Exception as e:
            self.connection.debug_print(f"node-connection::{self.id}::decompress::error::{e}")

        return data

    def send(self, data: bytes, compression: False):
        try:
            if compression:
                data = self.compress(data)
                if data is not None:
                    self.sock.sendall(data + self.COMPRESS_CHAR + self.EOT_CHAR)
            else:
                self.sock.sendall(data + self.EOT_CHAR)

            print(f"node-connection::{self.id}::sending::{len(data)}-bytes")

        except Exception as e:
            self.connection.debug_print(f"node-connection::{self.id}::send-data-error: {e}")
            self.stop()

    def parse_packet(self, packet) -> bytes:
        if packet.find(self.COMPRESS_CHAR) == len(packet) - 1:
            packet = self.decompress(packet[:-1])

        return packet

    def run(self):
        buffer = b""

        while not self.terminate_flag.is_set():
            chunk = b""

            try:
                chunk = self.sock.recv(4096)

            except socket.timeout:
                self.connection.debug_print(f"node-connection::{self.id}::timeout")

            except Exception as e:
                self.terminate_flag.set()
                self.connection.debug_print(f"node-connection::{self.id}::error: {e}")

            if chunk != b"":
                buffer += chunk
                eot_pos = buffer.find(self.EOT_CHAR)

                while eot_pos > 0:
                    packet = buffer[:eot_pos]
                    buffer = buffer[eot_pos + 1:]

                    self.connection.node_message(self, self.parse_packet(packet))

                    eot_pos = buffer.find(self.EOT_CHAR)

            time.sleep(0.01)

        self.sock.settimeout(None)
        self.sock.close()

        self.connection.node_disconnect(self)
        self.connection.debug_print(f"node-connection::{self.id}::stopped")

    def stop(self) -> None:
        self.terminate_flag.set()
