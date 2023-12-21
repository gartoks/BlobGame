import socket
import struct
import gzip

from Packets import *

class SocketController:
    def __init__(self, host_tuple, worker_id=0) -> None:
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self.connection.connect(host_tuple)
        self.worker_id = worker_id
        print(f"Connected worker {worker_id} to {host_tuple}")

        bytes = self.receive_exact(GAME_INFO_PACKET_SIZE)
        self.game_info = GameInfoPacket()
        
        (
            self.game_info.game_index,
            gamemode_key_bytes
        ) = struct.unpack_from("i20s", buffer=bytes)

        self.game_info.gamemode_key = gamemode_key_bytes.decode("utf8")

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
        bytes = gzip.decompress(self.receive_exact(size))

        frame = FramePacket()
        
        (
            frame.blob_count,
            frame.current_blob_type,
            frame.next_blob_type,
            frame.held_blob_type,
            frame.current_score,
            frame.can_spawn_blob,
            frame.is_game_over,
        ) = struct.unpack_from("iiiii??", bytes)

        blobsParts = struct.unpack_from("ffi"*frame.blob_count, buffer=bytes, offset=FRAME_PACKET_SIZE)
        frame.blobs = [Blob(blobsParts[i], blobsParts[i+1], blobsParts[i+2]) for i in range(0, len(blobsParts), 3)]

        return frame

