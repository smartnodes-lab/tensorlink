import base64
import gc
import hashlib
import logging
import os
import socket
import threading
import zlib
from datetime import datetime
from typing import Union

CHUNK_SIZE = 2048


def join_writing_threads(writing_threads):
    """Joins all writing threads."""
    for t in writing_threads:
        t.join()
    writing_threads.clear()


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
        self.ghosts = 0

        self.host = host
        self.port = port
        self.main_port = port
        self.main_node = main_node
        self.sock = sock
        self.terminate_flag = threading.Event()
        self.last_seen = None
        self.stats = {}

        self.node_key = node_key
        self.node_id = hashlib.sha256(node_key).hexdigest()
        self.role = role
        self.sock.settimeout(10)
        self.chunk_size = CHUNK_SIZE

        # End of transmission + compression characters for the network messages.
        self.EOT_CHAR = b"HELLOCHENQUI"
        self.COMPR_CHAR = 0x02.to_bytes(16, "big")

        # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)  # 4MB receive buffer
        # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32 * 1024 * 1024)  # 4MB send buffer

    def run(self):
        buffer = b""
        prefix = b""
        writing_threads = []

        while not self.terminate_flag.is_set():
            chunk = self.receive_chunk()
            if chunk is None:  # Indicates a disconnection or error
                break

            self.update_last_seen()

            if chunk:
                buffer, prefix = self.process_chunk(
                    chunk, buffer, prefix, writing_threads
                )

            gc.collect()

        self.cleanup()

    def receive_chunk(self):
        """Handles receiving a chunk of data from the socket."""
        try:
            return self.sock.recv(self.chunk_size)
        except socket.timeout:
            return None
        except (ConnectionResetError, ConnectionAbortedError) as e:
            self.handle_disconnection(f"Connection lost: {e}")
            return None
        except Exception as e:
            self.handle_unexpected_error(e)
            return None

    def process_chunk(self, chunk, buffer, prefix, writing_threads):
        """Processes a received chunk and updates the buffer and prefix."""
        if chunk.startswith(b"MODULE"):
            prefix = chunk[:70]  # MODULE + module_id
            buffer += chunk[70:]
        elif chunk.startswith(b"PARAMETERS"):
            prefix = chunk[:74]  # PARAMETERS + module_id
            buffer += chunk[74:]
        else:
            buffer += chunk

        buffer = self.handle_packet(buffer, prefix, writing_threads)
        return buffer, prefix

    def handle_packet(self, buffer, prefix, writing_threads):
        """Handles a complete packet if the EOT_CHAR is found."""
        eot_pos = buffer.find(self.EOT_CHAR)
        if eot_pos >= 0:
            packet = buffer[:eot_pos]
            self.write_packet_to_file(packet)
            buffer = buffer[eot_pos + len(self.EOT_CHAR) :]
            self.main_node.handle_message(self, b"DONE STREAM" + prefix)
            join_writing_threads(writing_threads)
        elif len(buffer) > 20_000_000:
            self.start_writing_thread(buffer, writing_threads)
            buffer = b""
        return buffer

    def write_packet_to_file(self, packet):
        """Writes a complete packet to the file."""
        file_name = self.get_file_name()
        try:
            with open(file_name, "ab") as f:
                f.write(packet)
        except Exception as e:
            self.main_node.debug_print(
                f"Connection -> file writing error: {e}",
                colour="bright_red",
                level=logging.ERROR,
            )

    def start_writing_thread(self, buffer, writing_threads):
        """Starts a new thread to write buffer data to a file."""
        file_name = self.get_file_name()
        t = threading.Thread(target=self.write_to_file, args=(file_name, buffer))
        writing_threads.append(t)
        t.start()

    def handle_disconnection(self, message):
        """Handles disconnection scenarios."""
        self.terminate_flag.set()
        self.main_node.debug_print(message, colour="bright_red", level=logging.ERROR)
        self.main_node.disconnect_node(self.node_id)

    def handle_unexpected_error(self, error):
        """Handles unexpected exceptions."""
        self.terminate_flag.set()
        self.main_node.debug_print(
            f"Connection -> unexpected error: {error}",
            colour="bright_red",
            level=logging.ERROR,
        )
        self.main_node.disconnect_node(self.node_id)

    def update_last_seen(self):
        """Updates the last seen timestamp."""
        self.last_seen = datetime.now()

    def cleanup(self):
        """Closes the socket and performs cleanup."""
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        finally:
            self.sock.settimeout(None)
            self.sock.close()

    def get_file_name(self):
        """Generates the file name for storing streamed data."""
        return f"tmp/streamed_data_{self.host}_{self.port}_{self.main_node.host}_{self.main_node.port}"

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
                num_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size
                data_view = memoryview(data)
                data_size = len(data)

                for chunk_number, i in enumerate(
                    range(0, data_size, self.chunk_size), start=1
                ):
                    self.sock.sendall(data_view[i : i + self.chunk_size])

                    # Print debug information for every chunk, or you can choose an interval.
                    if chunk_number % 100 == 0:
                        self.main_node.debug_print(
                            f"Connection -> Sent chunk {chunk_number} of {num_chunks}",
                            colour="magenta",
                        )

                self.sock.sendall(self.EOT_CHAR)

        except Exception as e:
            self.main_node.debug_print(
                f"Connection -> connection send error: {e}",
                colour="bright_red",
                level=logging.ERROR,
            )
            self.stop()

    def send_from_file(self, file_name: str, tag: bytes):
        try:
            # Data is a filename. Read from file in chunks.
            with open(file_name, "rb") as file:
                self.sock.sendall(tag)

                # Get the total file size
                total_size = os.fstat(file.fileno()).st_size
                chunk_size = self.chunk_size
                num_chunks = (total_size + chunk_size - 1) // chunk_size

                chunk_number = 0

                while True:
                    current_position = file.tell()
                    bytes_left = total_size - current_position

                    # Optionally print or log the number of bytes left
                    if chunk_number % 10 == 0:
                        self.main_node.debug_print(
                            f"Connection -> Bytes left to send: {bytes_left}",
                            colour="magenta",
                        )
                        self.main_node.debug_print(
                            f"Connection -> Sent chunk {chunk_number} of {num_chunks}",
                            colour="magenta",
                        )

                    chunk = file.read(chunk_size)
                    if not chunk:
                        break

                    self.sock.sendall(chunk)
                    chunk_number += 1

                    gc.collect()

                self.sock.sendall(self.EOT_CHAR)

            os.remove(file_name)

        except Exception as e:
            self.main_node.debug_print(
                f"Connection -> Error sending file: {e}",
                colour="bright_red",
                level=logging.ERROR,
            )
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

    def write_to_file(self, file_name, buffer):
        try:
            with open(file_name, "ab") as f:
                f.write(buffer)
        except Exception as e:
            self.main_node.debug_print(
                f"Connection -> file writing error: {e}",
                level=logging.ERROR,
                colour="bright_red",
            )

    def compress(self, data):
        compressed = data

        try:
            compressed = base64.b64encode(zlib.compress(data, 6))
        except Exception as e:
            self.main_node.debug_print(
                f"Connection -> compression-error: {e}",
                level=logging.CRITICAL,
                colour="bright_red",
            )

        return compressed

    def decompress(self, data):
        decompressed = base64.b64decode(data)

        try:
            decompressed = zlib.decompress(decompressed)
        except Exception as e:
            self.main_node.debug_print(
                f"Connection -> decompression-error: {e}",
                colour="bright_red",
                level=logging.CRITICAL,
            )

        return decompressed

    def adjust_chunk_size(self, chunk_size: str = None):
        if chunk_size == "large":
            self.chunk_size = CHUNK_SIZE**2
        else:
            self.chunk_size = CHUNK_SIZE
