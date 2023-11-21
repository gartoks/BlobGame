import socket
import struct

from FrameInfo import FrameInfo, Blob, BLOB_SIZE

class SocketController:
    def __init__(self, host_tuple, worker_id=0) -> None:
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.connection.connect(host_tuple)
        self.worker_id = worker_id
        print(f"Connected worker {worker_id} to {host_tuple}")

    def close_connection(self):
        self.connection.close()

    def send_frame_info(self, t: float, shouldDrop: bool):
        buffer = struct.pack("f?", t, shouldDrop)
        self.connection.send(buffer)
    
    def receive_exact(self, count: int):
        data = self.connection.recv(count)
        
        while (len(data) < count):
            data += self.connection.recv(count - len(data))

            if (len(data) == 0):
                raise socket.error("Connection closed")
        return data

    def receive_frame_info(self):
        count, = struct.unpack("i", self.receive_exact(4))
        
        bytes_left = BLOB_SIZE * count
        buffer = self.receive_exact(bytes_left)
        
        frame = FrameInfo([], -1, -1, False, 0, False, 0)
        for i in range(0, bytes_left, BLOB_SIZE):
            x, y, t = struct.unpack("ffi", buffer[i:i+BLOB_SIZE])
            frame.blobs.append(Blob(x, y, t))
        
        frame.current_blob, frame.next_blob, frame.score, frame.game_index, frame.can_drop, frame.is_game_over = \
            struct.unpack("iiii??", self.receive_exact(18))
        return frame

