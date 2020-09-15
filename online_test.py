
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

# Make an instance of the network
net = network.Network()


# Point cloud visualization:  https://github.com/heremaps/pptk

class SpeedSource(Enum):
    CONST = 1
    ROBOT = 2

zeros_one = np.matrix('0 0 0 1')

# Velocity camera
velocity_camera = [0.0, 0.0, 0.0]
R_robot = np.matrix('1 0 0; 0 1 0; 0 0 1')
R_robot_prev = np.matrix('1 0 0; 0 1 0; 0 0 1')
t_robot = np.matrix('0; 0; 0')
t_robot_prev = np.matrix('0; 0; 0')

T_base_tcp_1 = np.append(R_robot_prev.transpose(),\
                -1.0 * R_robot_prev.transpose().dot(t_robot_prev), axis = 1)
T_base_tcp_1 = np.append(T_base_tcp_1, zeros_one, axis = 0)

T_tcp_base_2 = np.append(R_robot, t_robot, axis = 1)
T_tcp_base_2 = np.append(T_tcp_base_2, zeros_one, axis = 0)


# Transformation between camera and TCP
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


# Algorithm settings
step = 16
threshold = 0.00045

speed_source = SpeedSource.CONST


if speed_source == SpeedSource.CONST:
    print("\nSpeed source CONST\n")
elif speed_source == SpeedSource.ROBOT:
    print("\nSpeed source ROBOT\n")


# Get config for the stream
config_1 = ri.get_config()


# Get pipeline for the stream
pipeline_1 = ri.get_pipeline()


# Enable SR300 stream
ri.enable_stream_sr300(config_1, 640, 480, 30)
dt = 1.0 / 30 #s

# Start pipeline
pipeline_1.start(config_1)


try:

    # Wait for a coherent pair of frames: depth and color
    frames_sr300 = ri.get_frames(pipeline_1)
    depth_frame = ri.get_depth_frames(frames_sr300)
    color_frame = ri.get_color_frames(frames_sr300)


    # Convert images to numpy arrays
    depth_image_prev = ri.convert_img_to_nparray(depth_frame)
    color_image_prev = ri.convert_img_to_nparray(color_frame)

    # Convert RGB image to gray (for optical flow)
    gray_prev = emf.rgb_to_gray(color_image_prev)
    ih,iw = gray_prev.shape


    # Align depth to color
    depth_frame_prev_aligned = emf.align_depth_to_color(frames_sr300)

    # Deprojection
    deprojected_prev = emf.deproject(depth_frame_prev_aligned, step=step)
    #deprojected_prev = emf.deproject(depth_frame_prev_aligned, 0, 479)


    #######################################################

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames_sr300 = ri.get_frames(pipeline_1)
        depth_frame = ri.get_depth_frames(frames_sr300)
        color_frame = ri.get_color_frames(frames_sr300)
        if not depth_frame or not color_frame:
            continue

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
        diff_flow = emf.flow_3d(deproject_flow_new, deproject_flow,dt)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = ri.get_depth_colormap(depth_image, 1)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        ri.show_imgs('Optical flow', emf.draw_flow(gray, flow.astype(int), step=step), 1)

        #######################################################


        velocities_from_egomotion = emf.velocity_from_point_clouds_online(deprojected, T_cam_tcp, T_tcp_cam, T_base_tcp_1, T_tcp_base_2, dt)


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

        print("")


    print("Executed succesfully.")


except (KeyboardInterrupt, SystemExit):
    print("Program exited.")



