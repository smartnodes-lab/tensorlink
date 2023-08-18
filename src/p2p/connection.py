from typing import Union
import socket
import time
import threading
import json
import zlib
import base64


class Connection(threading.Thread):
    def __init__(self, main_node, sock: socket.socket, id: str, host: str, port: int):
        super(Connection, self).__init__()

        self.host = host
        self.port = port
        self.main_node = main_node
        self.sock = sock
        self.terminate_flag = threading.Event()

        self.id = id
        self.sock.settimeout(5)

        # End of transmission + compression characters for the network messages.
        self.EOT_CHAR = 0x04.to_bytes(1, 'big')
        self.COMPR_CHAR = 0x02.to_bytes(1, 'big')

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
            print(f"Exception: {e}")

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

        try:
            packet_decoded = packet.decode('utf-8')

            try:
                return json.loads(packet_decoded)
            except json.decoder.JSONDecodeError:
                return packet_decoded

        except UnicodeDecodeError:
            return packet

    def run(self):
        buffer = b""

        while not self.terminate_flag.is_set():
            chunk = b""

            try:
                chunk = self.sock.recv(4096)
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

                    self.main_node.node_message(self, self.parse_packet(packet))

                    eot_pos = buffer.find(self.EOT_CHAR)

            time.sleep(0.01)

        self.sock.settimeout(None)
        self.sock.close()
