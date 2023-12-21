from SocketController import SocketController
from Renderer import Renderer
from Constants import *

import socket
import struct
import random

renderer = Renderer()

try:
    while True:
        try:
            controller = SocketController(("localhost", 1337))
            print(f"Game index {controller.game_info.game_index}, gamemode: {controller.game_info.gamemode_key}")

            t = -1
            shouldDrop = False
            frames_after_landing = 0
            while True:
                frame = controller.receive_frame_info()

                if (shouldDrop):
                    t = -1

                # sort from top to bottom
                frame.blobs.sort(key=lambda blob: blob.y)
                
                same_blob_types = list(filter(lambda blob: blob.type == frame.current_blob_type, frame.blobs))
                if (same_blob_types):
                    t = (same_blob_types[0].x + ARENA_WIDTH/2) / ARENA_WIDTH
                elif (t == -1):
                    t = random.random()

                if (frame.can_spawn_blob):
                    frames_after_landing += 1
                else:
                    frames_after_landing = 0

                # wait for .5 seconds to let things combine and move around
                shouldDrop = frames_after_landing > 30

                if (frame.is_game_over):
                    controller.close_connection()
                    controller = SocketController(("localhost", 1337))
                    continue

                controller.send_frame_info(t, shouldDrop, False)
                # renderer.render_frame(frame, t)
                # renderer.display_frame()


        except socket.error:
            print("Connection was closed. Restarting.")
        except struct.error:
            print("Invalid data was received. Restarting.")
        
        controller.close_connection()
except KeyboardInterrupt:
    pass
renderer.quit()
controller.close_connection()
