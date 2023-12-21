import socket
import struct

from FrameInfo import FramePacket, Blob, BLOB_SIZE

class SocketController:
    def __init__(self, host_tuple, worker_id=0) -> None:
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.connection.connect(host_tuple)
        self.worker_id = worker_id
        print(f"Connected worker {worker_id} to {host_tuple}")

    def close_connection(self):
        self.connection.close()

    def send_frame_info(self, t: float, shouldDrop: bool, shouldHold: bool):
        buffer = struct.pack("f??", t, shouldDrop, shouldHold)
        self.connection.send(buffer)
    
    def receive_exact(self, count: int):
        data = self.connection.recv(count)
        
        while (len(data) < count):
            data += self.connection.recv(count - len(data))

            if (len(data) == 0):
                raise socket.error("Connection closed")
        return data

    def receive_frame_info(self):
        size, = struct.unpack("i", self.receive_exact(4))
        bytes = self.receive_exact(size)

        frame = FramePacket()
        
        (
            frame.blob_count,
            frame.current_blob_type,
            frame.next_blob_type,
            frame.held_blob_type,
            frame.current_score,
            frame.game_index,
            frame.can_spawn_blob,
            frame.is_game_over,
        ) = struct.unpack_from("iiiiii??", bytes)

        return frame

