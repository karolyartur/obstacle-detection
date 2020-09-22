import zmq
import numpy as np
import matplotlib.pyplot as plt
import cv2
import Msg
import flatbuffers
import math  

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:1111")

while True:
    socket.subscribe(b'd\x02\x00\x03')
    topic = socket.recv()
    array_buffer = socket.recv()
    data_bytearray = bytearray(array_buffer)
    msg = Msg.Msg.GetRootAsMsg(array_buffer,0)
    
    msg_as_np = msg.ValueVectorAsNumpy()
    
    x_sensorfusion = msg_as_np[0]
    y_sensorfusion = msg_as_np[1]
    phi_sensorfusion = msg_as_np[2]
    v_sensorfusion = msg_as_np[3]
    omega_sensorfusion = msg_as_np[4]
    
    v_vehicle_0 = math.cos(phi_sensorfusion)*v_sensorfusion
    v_vehicle_1 = math.sin(phi_sensorfusion)*v_sensorfusion
    
    v_vehicle_sensorfusion = np.matrix([[v_vehicle_0],  [v_vehicle_1] , [0]])
    omega_vehicle_sensorfusion = np.matrix([[0],  [0] , [omega_sensorfusion]])

    print(v_vehicle_sensorfusion, omega_vehicle_sensorfusion)
    
    
    
    
    


    
    
    
    
    
    

    


