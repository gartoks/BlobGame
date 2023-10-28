from SocketController import SocketController
import time
import math
import socket

UPS = 55
TIME_SCALE = 1

controller = SocketController(("localhost", 1234))

controller.start_listening()

while True:
    try:
        controller.wait_for_connection()

        while True:
            controller.send_frame_info(abs(math.sin(time.time()*2 * TIME_SCALE)/2 + 0.5), True)

            time.sleep((1/UPS) / TIME_SCALE)
    except socket.error:
        print("Connection was closed. Restarting.")