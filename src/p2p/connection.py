import hashlib
from typing import Union
import socket
import time
import threading
import json
import zlib
import base64


class Connection(threading.Thread):
    def __init__(
        self,
        main_node,
        sock: socket.socket,
        host: str,
        port: int,
        main_port: int,
        node_key: bytes,
        role: int,
    ):
        super(Connection, self).__init__()
        self.ping = -1
        self.pinged = -1
        self.reputation = 50

        self.host = host
        self.port = port
        self.main_port = port
        self.main_node = main_node
        self.sock = sock
        self.terminate_flag = threading.Event()
        self.stats = {}

        self.node_key = node_key
        self.node_id = hashlib.sha256(node_key).hexdigest().encode()
        self.role = role
        self.sock.settimeout(60)
        self.chunk_size = 131_072

        # End of transmission + compression characters for the network messages.
        self.EOT_CHAR = b"HELLOCHENQUI"
        self.COMPR_CHAR = 0x02.to_bytes(16, "big")

    def compress(self, data):
        compressed = data

        try:
            compressed = base64.b64encode(zlib.compress(data, 6))
        except Exception as e:
            self.main_node.debug_print(f"compression-error: {e}")

        return compressed

    def decompress(self, data):
        decompressed = base64.b64decode(data)

        try:
            decompressed = zlib.decompress(decompressed)
        except Exception as e:
            self.main_node.debug_print(f"decompression-error: {e}")

        return decompressed

    def send(self, data: bytes, compression: bool = False):
        try:
            if compression:
                data = self.compress(data)
                if data is not None:
                    for i in range(0, len(data), self.chunk_size):
                        chunk = data[i : i + self.chunk_size]
                        self.sock.sendall(chunk + self.COMPR_CHAR)
                    self.sock.sendall(self.EOT_CHAR)
            elif len(data) < self.chunk_size:
                self.sock.sendall(data + self.EOT_CHAR)
            else:
                for i in range(0, len(data), self.chunk_size):
                    chunk = data[i : i + self.chunk_size]
                    self.sock.sendall(chunk)

                self.sock.sendall(self.EOT_CHAR)
        except Exception as e:
            self.main_node.debug_print(f"connection send error: {e}")
            self.stop()

    def stop(self) -> None:
        self.terminate_flag.set()

    def parse_packet(self, packet) -> Union[str, dict, bytes]:
        if packet.find(self.COMPR_CHAR) == len(packet) - 1:
            packet = self.decompress(packet[0:-1])

        return packet

        # try:
        #     packet_decoded = packet.decode()
        #
        #     try:
        #         return json.loads(packet_decoded)
        #     except json.decoder.JSONDecodeError:
        #         return packet_decoded
        #
        # except UnicodeDecodeError:
        #     return packet

    def run(self):
        buffer = b""
        b_size = 0

        while not self.terminate_flag.is_set():
            chunk = b""

            try:
                chunk = self.sock.recv(1_000_000)
            except socket.timeout:
                self.main_node.debug_print(f"connection timeout")
            except Exception as e:
                self.terminate_flag.set()
                self.main_node.debug_print(f"unexpected error: {e}")

            if chunk != b"":
                eot_pos = chunk.find(self.EOT_CHAR)
                file_name = f"streamed_data_{self.host}_{self.port}_{self.main_node.host}_{self.main_node.port}"

                # We have reached the end of one nodes processing
                if eot_pos > 0:
                    packet = buffer + chunk[:eot_pos]
                    try:
                        with open(file_name, "ab") as f:
                            f.write(packet)
                            buffer = b""
                            b_size = 0

                        self.main_node.handle_message(self, b"DONE STREAM")

                    except Exception as e:
                        raise e

                elif len(buffer) > 40_000_000:
                    try:
                        with open(file_name, "ab") as f:
                            b_size += len(buffer)
                            f.write(buffer)
                            buffer = chunk

                    except Exception as e:
                        raise e

                else:
                    buffer += chunk

        self.sock.settimeout(None)
        self.sock.close()

    """
    Connection thread between two nodes that are able to send/stream data from/to
    the connected node.

    TODO
        Send message size before to prepare accordingly.
        Switch between saving bytes to loading directly based on packet size.
    """
