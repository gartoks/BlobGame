import socket

from SocketController import SocketController

class SocketServer:
    def __init__(self, host_tuple: (str, int)) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.host_tuple = host_tuple
    
    def start_listening(self):
        print(f"Listening on {self.host_tuple[0]}:{self.host_tuple[1]}")
        self.server.bind(self.host_tuple)
        self.server.listen(1)

    def wait_for_connection(self):
        connection, addr = self.server.accept()
        print(f"Connection from {addr}")
        return SocketController(connection, addr)

    def close_server(self):
        self.server.close()