## @package obstacle-detection
#  Test obstacle-detection ZMQ server
#
#  This module can be used to test the connection to the obstacle-detection ZMQ server by making a client that subscribes to the obstacle-detection result topic

import zmq
import numpy as np
import matplotlib.pyplot as plt
import cv2

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.subscribe("")
while True:
    obstacle_width_center = socket.recv_pyobj()
    print(obstacle_width_center)
