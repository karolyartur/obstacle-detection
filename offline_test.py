
import pyrealsense2 as rs
import numpy as np
import cv2
import realsense_input_module as ri
import realsense_intrinsics_module as intr
import egomotion_filter_module as emf
import matplotlib.pyplot as plt
import read_robot_states
import time
import sys
import pptk
import ur_tests_module as tests
from mpl_toolkits import mplot3d
from numpy.polynomial.polynomial import polyfit
from enum import Enum
import ur_rotation_module as urrot
import numpy as np
import cv2 as cv
import network
import utils
import math
from time import sleep
import zmq



# ZMQ init
context = zmq.Context()
socket = context.socket(zmq.PUB)
print('Binding to port 5555')
socket.bind("tcp://*:5555")
sleep(0.033)

# Make an instance of the network
net = network.Network()


# Point cloud visualization:  https://github.com/heremaps/pptk

class SpeedSource(Enum):
    CONST = 1
    ROBOT = 2
    T265 = 3

# Crop data to cut the acceleration
time_crop_point_start = 0.5 * 1000.0
time_crop_point_end = 30 * 1000.0

# Velocity camera
velocity_camera = [-0.08, 0.0, 0.0]



# Transformation between camera and TCP
zeros_one = np.matrix('0 0 0 1')

R_cam_tcp = np.matrix('-1 0 0; 0 0 -1; 0 -1 0')
t_cam_tcp = np.matrix('-0.025; -0.053; 0.058')


T_cam_tcp = np.append(R_cam_tcp, t_cam_tcp, axis = 1)
T_cam_tcp = np.append(T_cam_tcp, zeros_one, axis = 0)
#print(T_cam_tcp)

R_tcp_cam = R_cam_tcp.transpose()
t_tcp_cam = -1.0 * R_tcp_cam.dot(t_cam_tcp)
T_tcp_cam = np.append(R_tcp_cam, t_tcp_cam, axis = 1)
T_tcp_cam = np.append(T_tcp_cam, zeros_one, axis = 0)
#print(T_tcp_cam)


# Transformation between the two robots

R_base_base_ball = np.matrix('-1 0 0; 0 -1 0; 0 0 1')
t_base_base_ball = np.matrix('-0.6565; -0.0035; 0.0')


T_base_base_ball = np.append(R_base_base_ball, t_base_base_ball, axis = 1)
T_base_base_ball = np.append(T_base_base_ball, zeros_one, axis = 0)

# Camera settings
width_sr300 = 640
height_sr300 = 480
framerate_sr300 = 30

# Algorithm settings
step = 16
threshold = 0.045
minRadius=40
maxRadius=65 #65 #75
radiusOffset=8
speed_source = SpeedSource.ROBOT

# ZMQ switch
is_zmq = True

if speed_source == SpeedSource.CONST:
    print("\nSpeed source CONST\n")
elif speed_source == SpeedSource.ROBOT:
    print("\nSpeed source ROBOT\n")
elif speed_source == SpeedSource.T265:
    print("\nSpeed source T265\n")


# Choose the data folder
if (len(sys.argv)<1 ):
    print("Usage details: python egomotion_filter:offline_sync_robot_ball.py path")
    sys.exit()

# Read the robot states
robot_states = read_robot_states.read_robot_states(str(sys.argv[1]) + "/" + "robot_states.txt")
robot_states_ball = read_robot_states.read_robot_states(str(sys.argv[1]) + "/" + "robot_states_ball.txt")


# Get config for the stream
config_1 = ri.get_config()
config_2 = ri.get_config()

# Get pipeline for the stream
pipeline_1 = ri.get_pipeline()
pipeline_2 = ri.get_pipeline()

# Start the cameras
profile_1 = tests.start_t265(config_1, pipeline_1, "t265.bag")
profile_2 = tests.start_sr300(config_2, pipeline_2, width_sr300, height_sr300, framerate_sr300, "sr300.bag")

try:

    # Wait for a coherent pair of frames: depth and color
    frames_sr300 = ri.get_frames(pipeline_2)
    depth_frame_prev = ri.get_depth_frames(frames_sr300)
    color_frame_prev = ri.get_color_frames(frames_sr300)

    # IMU
    frames_t265 = ri.get_frames(pipeline_1)

    # Sync start times, in msec
    start_time_imu = ri.get_pose_frames(frames_t265).get_timestamp()
    start_time_depth = depth_frame_prev.get_timestamp()
    start_time_color = color_frame_prev.get_timestamp()
    start_time_robot = robot_states[0].timestamp
    start_time_robot_ball = robot_states_ball[0].timestamp

    while ((depth_frame_prev.get_timestamp() - start_time_depth) < time_crop_point_start):
            frames_sr300 = ri.get_frames(pipeline_2)
            depth_frame_prev = ri.get_depth_frames(frames_sr300)
            color_frame_prev = ri.get_color_frames(frames_sr300)

    # Convert images to numpy arrays
    depth_image_prev = ri.convert_img_to_nparray(depth_frame_prev)
    color_image_prev = ri.convert_img_to_nparray(color_frame_prev)

    # Convert RGB image to gray (for optical flow)
    gray_prev = emf.rgb_to_gray(color_image_prev)
    ih,iw = gray_prev.shape


    # Align depth to color
    depth_frame_prev_aligned = emf.align_depth_to_color(frames_sr300)

    # Deprojection
    deprojected_prev = emf.deproject(depth_frame_prev_aligned, step=step)
    #deprojected_prev = emf.deproject(depth_frame_prev_aligned, 0, 479)

    frame_number_color_prev = color_frame_prev.get_frame_number()
    frame_number_depth_prev = depth_frame_prev_aligned.get_frame_number()
    current_time_prev = depth_frame_prev.get_timestamp() - start_time_depth

    robot_i_prev = 0;
    robot_ball_i_prev = 0;


    #######################################################

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames_sr300 = ri.get_frames(pipeline_2)
        depth_frame = ri.get_depth_frames(frames_sr300)
        color_frame = ri.get_color_frames(frames_sr300)
        frame_number_color = color_frame.get_frame_number()
        frame_number_depth = depth_frame.get_frame_number()

        #######################################################
        # Sync
        if ((frame_number_depth <= frame_number_depth_prev) or (
                frame_number_color <= frame_number_color_prev)) or (
                current_time_prev > time_crop_point_end):
            break;


        time_depth = depth_frame.get_timestamp() - start_time_depth
        time_color = color_frame.get_timestamp() - start_time_depth
        current_time = time_depth

        # Sync IMU
        frames_t265, time_imu =  tests.sync_imu(ri, pipeline_1, frames_t265, start_time_imu, current_time, 5.5)

        # Sync robot
        robot_i, time_robot = \
            tests.sync_robot_state(robot_states, start_time_imu, robot_i_prev, current_time, 32.0)
        print("Time robot:\t{} [ms]".format(time_robot))
        velocity_robot = \
            read_robot_states.get_velocity(robot_states[robot_i],
                                   robot_states[robot_i_prev])
        robot_dt = \
            read_robot_states.get_dt(robot_states[robot_i], robot_states[robot_i_prev])
        r_robot = robot_states[robot_i].get_rotation_vector()
        r_robot_prev = robot_states[robot_i_prev].get_rotation_vector()
        print("Velocity robot:\t\t{} [m/s]".format(velocity_robot))

        # Sync robot ball
        robot_ball_i, time_robot_ball = \
            tests.sync_robot_state(robot_states_ball, start_time_imu, robot_ball_i_prev, current_time, 32.0)
        print("Time robot ball:\t{} [ms]".format(time_robot_ball))
        velocity_robot_ball = \
            read_robot_states.get_velocity(robot_states_ball[robot_ball_i],
                                   robot_states_ball[robot_ball_i_prev])
        velocity_robot_ball = tests.velocity_ball_to_camera_frame( \
            velocity_robot_ball, T_cam_tcp, T_base_base_ball, robot_states[robot_i])
        print("Velocity robot ball:\t{} [m/s]".format(velocity_robot_ball))

        #######################################################

        # Read images
        depth_image = ri.convert_img_to_nparray(depth_frame)
        color_image = ri.convert_img_to_nparray(color_frame)



        # Convert RGB image to gray (for optical flow)
        gray = emf.rgb_to_gray(color_image)

        #######################################################



        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray,
                            None, 0.5, 5, 15, 3, 5, 1,
                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        # Get the previous and the current pixel locations
        lines = emf.get_lines(gray, flow, step=step)

        # Align depth to color
        depth_frame_aligned = emf.align_depth_to_color(frames_sr300)

        # Deprjection
        deprojected = emf.deproject(depth_frame_aligned, step=step)
        #deprojected = emf.deproject(depth_frame_aligned, 0, 479)

        # Deproject the previous pixels for 3D optical flow
        deproject_flow = emf.deproject_flow_prev(depth_frame_prev_aligned,
                                         lines, step=step)

        # Deproject the current pixels for 3D optical flow
        deproject_flow_new = emf.deproject_flow_new(depth_frame_aligned, lines, step=step)

        # Calculate 3D optical flow
        diff_flow = emf.flow_3d(deproject_flow_new, deproject_flow,robot_dt)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = ri.get_depth_colormap(depth_image, 1)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        ri.show_imgs('Optical flow', emf.draw_flow(gray, flow.astype(int), step=step), 1)

        #######################################################

        # Get pose data from RealSense
        pose = ri.get_pose_frames(frames_t265)
        if pose:
            pose_data = ri.get_pose_data(pose)
            velocity = ri.get_velocity(pose_data)
            angular_velocity = ri.get_angular_velocity(pose_data)
            position = ri.get_translation(pose_data)




        # Calc rotation matrices for the robot
        R_robot=urrot.rot_vec2rot_mat(r_robot)
        R_robot_prev = urrot.rot_vec2rot_mat(r_robot_prev)

        velocities_from_egomotion = emf.velocity_from_point_clouds(deprojected, T_cam_tcp, T_tcp_cam, robot_states[robot_i_prev], robot_states[robot_i], robot_dt)


        # Compare the velocities
        egomotion_filtered_flow = emf.velocity_comparison(depth_frame_aligned, diff_flow, velocities_from_egomotion,threshold, step=step)

        nonzero_elements = egomotion_filtered_flow[np.nonzero(egomotion_filtered_flow > 0)]
        nonzero_indices = np.where(egomotion_filtered_flow != 0)[0]

        filtered_to_flow = emf.filtered_flow_2d(egomotion_filtered_flow, flow, step=step)
        ri.show_imgs('Optical flow filtered', emf.draw_flow(gray, filtered_to_flow, step=step), 1)

        full_len = deproject_flow_new.shape[0]*deproject_flow_new.shape[1]

        deproject_flow_new_flat = deproject_flow_new.reshape(full_len,3)
        three_d_flow_x = np.squeeze(diff_flow[:,:,0].reshape(full_len,1))
        three_d_flow_x = three_d_flow_x

        #######################################################
        # Prepare data for network

        use_filtered_flow = True

        if (use_filtered_flow):
            flow_for_nn=filtered_to_flow[:,int((iw-ih)/2):int((iw+ih)/2)]
        else:
            flow_for_nn=flow[:,int((iw-ih)/2):int((iw+ih)/2)]

        flow_for_nn = cv.resize(flow_for_nn, (299,299))

        gray_resized=gray[:,int((iw-ih)/2):int((iw+ih)/2)]
        gray_resized = cv.resize(gray_resized, (299,299))

        #######################################################
        # Obstacle detection

        conc_img = utils.form_network_input(gray_resized, flow_for_nn)

        # Make predictions with the network
        prediction,mask = net.predict(conc_img)

        # Visualization
        viz = cv.resize(np.reshape(mask,(30,30)), (299,299))
        mask_small = np.reshape(mask,(30,30))
        #mask_small = np.zeros((30,30)) # for validation
        viz_2 = cv.resize(np.reshape(prediction,(30,30)), (299,299))
        utils.visualize_flow(viz)
        utils.visualize_flow(viz_2, name='pred')
        utils.visualize_flow(gray_resized, name='imgs')


        #######################################################

        # Bounding box

        mask_small_blob = mask_small.astype(np.uint8)
        mask_small_blob = emf.find_largest_blob(mask_small_blob)
        utils.visualize_flow(np.array(cv.resize(mask_small_blob, (299,299)), dtype=np.float32), name='mask_small_blob')
        blob_max = mask_small_blob.reshape((mask_small_blob.shape[0]*mask_small_blob.shape[1])).max(axis=0)

        if (blob_max > 0):
            bh, bw, bd = deprojected.shape
            deprojected_coordinates_robot_small=deprojected[:,int((bw-bh)/2):int((bw+bh)/2),:]
            bb = emf.get_3D_bounding_box(deprojected_coordinates_robot_small, mask_small_blob)
            # Width estimation
            blob_avg_coords = emf.avg_coords(deprojected_coordinates_robot_small, mask_small_blob)
            blob_width = emf.max_blob_width(deprojected_coordinates_robot_small, mask_small_blob, depth_frame_aligned, blob_avg_coords, step = step)
            blob_send = np.array([blob_avg_coords, blob_width])
            # Width estimation end
            print("Bounding box:\n({}\t{}\t{})\n({}\t{}\t{})".format( \
                        bb.get('x1'), bb.get('y1'), bb.get('z1'), \
                        bb.get('x2'), bb.get('y2'), bb.get('z2')))
        else:
            bb = {'x1': math.nan, 'y1': math.nan, 'z1': math.nan,\
                  'x2': math.nan, 'y2': math.nan, 'z2': math.nan}
                      

        #######################################################
        
        # ZMQ send object center and size
        
        if is_zmq:
            #message = socket.recv_string()
            #if message == "read":
            socket.send_pyobj(blob_send)
        
        #######################################################

        # Calculate props of the moving object
        eh, ew, ed = egomotion_filtered_flow.shape
        egomotion_filtered_flow_masked=egomotion_filtered_flow[:,\
                                            int((ew-eh)/2):int((ew+eh)/2),:]

        # Mask result on point velocities
        egomotion_filtered_flow_masked = np.multiply(egomotion_filtered_flow_masked,\
                                np.stack((mask_small, mask_small, mask_small), axis=2))

        velocity_mean_nonzero_elements, velocity_std_nonzero_elements = \
                            emf.calc_mean_velocity(egomotion_filtered_flow_masked)
        print("Result velocity mean:\t{} [m/s]"\
                    .format(velocity_mean_nonzero_elements))
        print("Result velocity std:\t{} [m/s]"\
                    .format(velocity_std_nonzero_elements))

        deproject_flow_new_masked=deproject_flow_new[:,\
                                    int((ew-eh)/2):int((ew+eh)/2),:]

        # Mask result on point positions
        deproject_flow_new_masked = np.multiply(deproject_flow_new_masked,\
                        np.stack((mask_small, mask_small, mask_small), axis=2))
        depth_mean_z, depth_std_z = \
                emf.calc_mean_depth(egomotion_filtered_flow_masked, \
                                        deproject_flow_new_masked)
        print("Result depth mean:\t{} [m]".format(depth_mean_z))
        print("Result depth std:\t{} [m]".format(depth_std_z))


        #######################################################

        # Because of optical flow we have to change the images
        gray_prev = gray

        # Change the points as well
        deprojected_prev = deprojected
        depth_frame_prev_aligned = depth_frame_aligned
        frame_number_color_prev = frame_number_color
        frame_number_depth_prev = frame_number_depth
        robot_i_prev = robot_i
        robot_ball_i_prev = robot_ball_i


        print("")


    print("Executed succesfully.")


except (KeyboardInterrupt, SystemExit):
    print("Program exited.")



