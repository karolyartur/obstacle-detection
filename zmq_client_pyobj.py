import zmq
import numpy as np
import matplotlib.pyplot as plt
import cv2

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
while True:
    socket.send_string('read')
    obstacle_width_center = socket.recv_pyobj()
    print(obstacle_width_center)
