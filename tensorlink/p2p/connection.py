from datetime import datetime
import threading
import logging
import hashlib
import socket
import os
import gc

CHUNK_SIZE = 2048


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
        """
        Initialize a network connection thread with peer and connection details.

        Args:
            main_node: The primary node managing this connection
            sock: Socket for communication
            host: Remote host address
            port: Remote host port
            main_port: Local main port
            node_key: Unique identifier for the remote node
            role: Connection role/type
        """
        super(Connection, self).__init__()
        # Connection tracking
        self.ping = -1
        self.pinged = -1

        # Peer connection metadata
        self.host = host
        self.port = port
        self.sock = sock
        self.node_key = node_key
        self.node_id = hashlib.sha256(node_key).hexdigest()
        self.role = role

        # Node and connection settings
        self.main_port = main_port
        self.main_node = main_node
        self.reputation = 50  # Unused reputation tracking

        # Connection state management
        self.terminate_flag = threading.Event()
        self.last_seen = None
        self.ghosts = 0  # Number of ghost transmissions (unexpected data)

        # Network configuration
        self.sock.settimeout(10)
        self.chunk_size = CHUNK_SIZE

        # Transmission markers
        self.EOT_CHAR = b"HELLOCHENQUI"  # End of Transmission marker
        self.COMPR_CHAR = 0x02.to_bytes(16, "big")  # Compression marker

        # Potential for optimizing data transmission
        # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)  # 4MB receive buffer
        # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32 * 1024 * 1024)  # 4MB send buffer

    def run(self):
        """
        Main connection thread method to handle data streaming and connection lifecycle.
        Manages receiving, buffering, and writing network data chunks.
        """
        buffer = b""
        prefix = b""
        writing_threads = []

        while not self.terminate_flag.is_set():
            try:
                chunk = self.sock.recv(self.chunk_size)
            except socket.timeout:
                continue
            except (ConnectionResetError, ConnectionAbortedError) as e:
                self._handle_connection_error(e)
                break
            except Exception as e:
                self._handle_unexpected_error(e)
                break

            if chunk:
                self.last_seen = datetime.now()
                (
                    buffer,
                    prefix,
                    is_transmission_complete,
                ) = self._process_data_chunk(chunk, buffer, prefix)

                # Manage large buffers by writing to file
                if len(buffer) > 20_000_000:
                    writing_thread = threading.Thread(
                        target=self._write_to_file,
                        args=(
                            f"tmp/streamed_data_{self.host}_{self.port}",
                            buffer,
                        ),
                    )
                    writing_threads.append(writing_thread)
                    writing_thread.start()
                    buffer = b""

            elif chunk == b"":
                self._handle_connection_close()
                break

            gc.collect()

        self._cleanup_socket()

    def send(self, data: bytes):
        """Send bytes to node"""
        try:
            if len(data) < self.chunk_size:
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
        """Send bytes from an existing file and delete it"""
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

    def _process_data_chunk(self, chunk: bytes, buffer: bytes, prefix: bytes) -> tuple:
        """
        Process an incoming data chunk and manage buffer and file writing.

        Args:
            chunk: Received network chunk
            buffer: Current data buffer
            prefix: Packet prefix (module/parameters)

        Returns:
            Updated buffer, prefix, and a flag indicating if processing is complete
        """
        file_name = f"tmp/streamed_data_{self.host}_{self.port}_{self.main_node.host}_{self.main_node.port}"

        # Handle special packet types with prefixes
        if b"MODULE" == chunk[:6]:
            prefix = chunk[:70]
            buffer += chunk[70:]
        elif b"PARAMETERS" == chunk[:10]:
            prefix = chunk[:74]
            buffer += chunk[74:]
        else:
            buffer += chunk

        # Check for end of transmission
        eot_pos = buffer.find(self.EOT_CHAR)
        if eot_pos >= 0:
            try:
                with open(file_name, "ab") as f:
                    f.write(buffer[:eot_pos])
            except Exception as e:
                self.main_node.debug_print(
                    f"File writing error: {e}",
                    colour="bright_red",
                    level=logging.ERROR,
                )

            # Signal transmission completion
            self.main_node.handle_message(self, b"DONE STREAM" + prefix)
            return buffer[eot_pos + len(self.EOT_CHAR) :], b"", True

        return buffer, prefix, False

    def _handle_connection_error(self, error):
        """Handle standard connection errors."""
        self.terminate_flag.set()
        self.main_node.debug_print(
            f"Connection -> Connection with {self.host}:{self.port} lost: {error}",
            colour="bright_red",
            level=logging.ERROR,
        )
        self.main_node.disconnect_node(self.node_id)

    def _handle_unexpected_error(self, error):
        """Handle unexpected connection errors."""
        self.terminate_flag.set()
        self.main_node.debug_print(
            f"Connection -> Unexpected connection error: {error}",
            colour="bright_red",
            level=logging.ERROR,
        )
        self.main_node.disconnect_node(self.node_id)

    def _handle_connection_close(self):
        """Handle graceful connection closure."""
        self.terminate_flag.set()
        self.main_node.debug_print(
            "Connection -> Connection closed.",
            colour="bright_red",
            level=logging.INFO,
        )
        self.main_node.disconnect_node(self.node_id)

    def _write_to_file(self, file_name, buffer):
        try:
            with open(file_name, "ab") as f:
                f.write(buffer)
        except Exception as e:
            self.main_node.debug_print(
                f"Connection -> file writing error: {e}",
                level=logging.ERROR,
                colour="bright_red",
            )

    def _cleanup_socket(self):
        """Perform final socket cleanup operations."""
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        finally:
            self.sock.settimeout(None)
            self.sock.close()

    def adjust_chunk_size(self, chunk_size: str = None):
        """
        Method to be implemented that adjusts the chunk size for higher reputation jobs / only during
        certain parts of the distributed training or proof of works (ie sending large models) TODO
        """
        if chunk_size == "large":
            self.chunk_size = CHUNK_SIZE**2
        else:
            self.chunk_size = CHUNK_SIZE
