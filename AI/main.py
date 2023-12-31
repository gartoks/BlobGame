from SocketController import SocketController
from Renderer import Renderer
from Constants import *

import socket
import struct
import random

controller = SocketController(("localhost", 1234))
renderer = Renderer((ARENA_WIDTH, ARENA_HEIGHT))


controller.start_listening()

try:
    while True:
        try:
            controller.wait_for_connection()

            t = -1
            shouldDrop = False
            frames_after_landing = 0
            while True:
                frame = controller.receive_frame_info()

                if (shouldDrop):
                    t = -1

                for i, blob in enumerate(frame.blobs):
                    frame.blobs[i].x += ARENA_OFFSET_X

                # sort from top to bottom
                frame.blobs.sort(key=lambda blob: blob.y)
                
                same_blob_types = list(filter(lambda blob: blob.type == frame.current_blob, frame.blobs))
                if (same_blob_types):
                    t = same_blob_types[0].x / ARENA_WIDTH
                elif (t == -1):
                    t = random.random()

                if (frame.can_drop):
                    frames_after_landing += 1
                else:
                    frames_after_landing = 0

                # wait for .5 seconds to let things combine and move around
                shouldDrop = frames_after_landing > 30

                controller.send_frame_info(t, shouldDrop)
                renderer.render_frame(frame)


        except socket.error:
            print("Connection was closed. Restarting.")
        except struct.error:
            print("Invalid data was received. Restarting.")
        
        controller.close_connection()
except KeyboardInterrupt:
    renderer.close_window()