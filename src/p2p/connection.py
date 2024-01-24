from typing import Union
import socket
import time
import threading
import json
import zlib
import base64


class Connection(threading.Thread):
    """
    Connection thread between two nodes that are able to send/stream data from/to
    the connected node.
    """
    def __init__(self, main_node, sock: socket.socket, id: str, host: str, port: int):
        super(Connection, self).__init__()

        self.host = host
        self.port = port
        self.main_node = main_node
        self.sock = sock
        self.terminate_flag = threading.Event()

        self.id = id
        self.sock.settimeout(3)
        self.latency = 0

        # End of transmission + compression characters for the network messages.
        self.EOT_CHAR = 0x03.to_bytes(4, 'big')
        self.COMPR_CHAR = 0x02.to_bytes(4, 'big')

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
                    self.sock.sendall(data + self.COMPR_CHAR + self.EOT_CHAR)
            else:
                self.sock.sendall(data + self.EOT_CHAR)
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

        while not self.terminate_flag.is_set():
            chunk = b""

            try:
                chunk = self.sock.recv(10_000)
            except socket.timeout:
                self.main_node.debug_print(f"connection timeout")
            except Exception as e:
                self.terminate_flag.set()
                self.main_node.debug_print(f"unexpected error: {e}")



            if chunk != b"":
                buffer += chunk
                eot_pos = buffer.find(self.EOT_CHAR)

                while eot_pos > 0:
                    packet = buffer[:eot_pos]
                    buffer = buffer[eot_pos + 1:]
                    eot_pos = buffer.find(self.EOT_CHAR)
                    self.main_node.handle_message(self, self.parse_packet(packet))

            time.sleep(0.001)

        self.sock.settimeout(None)
        self.sock.close()

    def measure_latency(self) -> float:
        start_time = time.time()
        self.send(b"LATENCY_TEST")
        self.sock.recv(128)
        end_time = time.time()
        return end_time - start_time

