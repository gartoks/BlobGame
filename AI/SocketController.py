import socket
import struct

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