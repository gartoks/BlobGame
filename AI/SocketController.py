import socket
import struct

from FrameInfo import FrameInfo, Blob, BLOB_SIZE

class SocketController:
    def __init__(self, host_tuple: (str, int)) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.host_tuple = host_tuple
        self.connection = None
    
    def start_listening(self):
        print(f"Listening on {self.host_tuple[0]}:{self.host_tuple[1]}")
        self.server.bind(self.host_tuple)
        self.server.listen(1)


    def wait_for_connection(self):
        self.connection, addr = self.server.accept()
        print(f"Connection from {addr}")

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
        
        frame = FrameInfo([], -1, -1, False)
        for i in range(0, bytes_left, BLOB_SIZE):
            x, y, t = struct.unpack("ffi", buffer[i:i+BLOB_SIZE])
            frame.blobs.append(Blob(x, y, t))
        
        frame.current_blob, frame.next_blob, frame.can_drop = struct.unpack("ii?", self.receive_exact(9))
        return frame

