
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
import yaml
import math
import zmq
import Msg
import flatbuffers
from time import sleep

##########################################################
# Sensorfusion subscribe ZMQ init

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:1111")

###########################################################

##########################################################
# Publish blob ZMQ init

context_p = zmq.Context()
socket_p = context.socket(zmq.PUB)
socket_p.bind("tcp://*:5555")

###########################################################
       
    
def run_obstacle_detection(registration_file):

    # Make an instance of the network
    net = network.Network()


    # Point cloud visualization:  https://github.com/heremaps/pptk

    class SpeedSource(Enum):
        CONST = 1
        ROBOT = 2

    zeros_one = np.matrix('0 0 0 1')

    # Velocity robot
    v_robot_const = np.matrix('0; 0; 0')
    omega_robot_const = np.matrix('0; 0; 0')

    with open(registration_file) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        reg_params = yaml.load(file, Loader=yaml.FullLoader)
        R_cam_robot = np.matrix(reg_params.get('R_cam_robot'))
        t_cam_robot = np.matrix(reg_params.get('t_cam_robot'))
    print("Camera registration read succesfully!\nR_cam_robot:\n{}\nt_cam_robot:\n{}".format(R_cam_robot,t_cam_robot))


    T_cam_robot = np.append(R_cam_robot, t_cam_robot, axis = 1)
    T_cam_robot = np.append(T_cam_robot, zeros_one, axis = 0)
    #print(T_cam_robot)

    R_robot_cam = R_cam_robot.transpose()
    t_robot_cam = -1.0 * R_robot_cam.dot(t_cam_robot)
    T_robot_cam = np.append(R_robot_cam, t_robot_cam, axis = 1)
    T_robot_cam = np.append(T_robot_cam, zeros_one, axis = 0)
    #print(T_robot_cam)


    # Algorithm settings
    step = 16
    threshold = 0.001

    speed_source = SpeedSource.ROBOT


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
        deproject_flow_prev = emf.deproject(depth_frame_prev_aligned, step=step)
        #deprojected_prev = emf.deproject(depth_frame_prev_aligned, 0, 479)

        # Visualize point cloud
        v = pptk.viewer(deproject_flow_prev)
        v.color_map('cool')
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


            # Deproject the current pixels for 3D optical flow
            deproject_flow = emf.deproject_flow_new(depth_frame_aligned, lines, step=step)

            # Calculate 3D optical flow
            diff_flow = emf.flow_3d(deproject_flow, deproject_flow_prev,dt)
            diff_flow_robot = emf.transform_velocites(diff_flow, T_robot_cam)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = ri.get_depth_colormap(depth_image, 1)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            ri.show_imgs('Optical flow', emf.draw_flow(gray, flow.astype(int), step=step), 1)

            #######################################################

            socket.subscribe(b'd\x02\x00\x03')
            if socket.recv() == True:
                speed_source == SpeedSource.ROBOT
            else:
                speed_source == SpeedSource.CONST

            if speed_source == SpeedSource.CONST:
                v_robot = v_robot_const
                omega_robot = omega_robot_const
            elif speed_source == SpeedSource.ROBOT:
                #topic = socket.recv()
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
    
                v_robot = np.matrix([[v_vehicle_0],  [v_vehicle_1] , [0]])
                omega_robot = np.matrix([[0],  [0] , [omega_sensorfusion]])

                print(v_robot)


            deprojected_coordinates_robot = emf.transform_points(deproject_flow, T_robot_cam)
            velocities_from_egomotion_robot = \
                emf.velocity_from_point_clouds_robot_frame(deprojected_coordinates_robot, \
                                            v_robot, omega_robot)


            # Compare the velocities
            egomotion_filtered_flow = \
                emf.velocity_comparison(depth_frame_aligned, \
                                            diff_flow_robot, \
                                            velocities_from_egomotion_robot, \
                                            threshold, step=step)

            nonzero_elements = egomotion_filtered_flow[np.nonzero(\
                                        egomotion_filtered_flow > 0)]
            nonzero_indices = np.where(egomotion_filtered_flow != 0)[0]

            filtered_to_flow = emf.filtered_flow_2d(egomotion_filtered_flow, \
                                        flow, step=step)
            ri.show_imgs('Optical flow filtered', \
                            emf.draw_flow(gray, filtered_to_flow, step=step), 1)

            full_len = deproject_flow.shape[0]*deproject_flow.shape[1]

            deproject_flow_flat = deproject_flow.reshape(full_len,3)
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
                bh, bw, bd = deprojected_coordinates_robot.shape
                deprojected_coordinates_robot_small=deprojected_coordinates_robot[:,int((bw-bh)/2):int((bw+bh)/2),:]
                bb = emf.get_3D_bounding_box(deprojected_coordinates_robot_small, mask_small_blob)
                
                blob_avg_coords = emf.avg_coords(deprojected_coordinates_robot_small, mask_small_blob)
                blob_width = emf.max_blob_width(deprojected_coordinates_robot_small, mask_small_blob, depth_frame_aligned, blob_avg_coords, step = step)
                blob_send = [blob_avg_coords, blob_width]
                print("Bounding box:\n({}\t{}\t{})\n({}\t{}\t{})".format( \
                        bb.get('x1'), bb.get('y1'), bb.get('z1'), \
                        bb.get('x2'), bb.get('y2'), bb.get('z2')))
            else:
                bb = {'x1': math.nan, 'y1': math.nan, 'z1': math.nan,\
                      'x2': math.nan, 'y2': math.nan, 'z2': math.nan}
                blob_send = [0, 0, 0, 0]  
            


            # Display point cloud
            emf.show_pointcloud(v, deproject_flow_flat, three_d_flow_x)

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

            deproject_flow_masked=deproject_flow[:,\
                                    int((ew-eh)/2):int((ew+eh)/2),:]

            # Mask result on point positions
            deproject_flow_masked = np.multiply(deproject_flow_masked,\
                        np.stack((mask_small, mask_small, mask_small), axis=2))
            depth_mean_z, depth_std_z = \
                emf.calc_mean_depth(egomotion_filtered_flow_masked, \
                                        deproject_flow_masked)
            print("Result depth mean:\t{} [m]".format(depth_mean_z))
            print("Result depth std:\t{} [m]".format(depth_std_z))



            #######################################################
            # ZMQ publish blob
            blob_msg_full = [blob_send, velocity_mean_nonzero_elements]
            socket_p.send_pyobj(blob_msg_full)

            #######################################################

            # Because of optical flow we have to change the images
            gray_prev = gray

            # Change the points as well
            deproject_flow_prev = deproject_flow
            depth_frame_prev_aligned = depth_frame_aligned

            print("")


        print("Executed succesfully.")


    except (KeyboardInterrupt, SystemExit):
        print("Program exited.")

    finally:
        # Stop streaming
        pipeline_1.stop()




if __name__ == "__main__":

    # Choose the data folder
    if (len(sys.argv)<2 ):
        print("Usage details: python online_test.py <path_for_registration_yaml_file>")
        sys.exit()

    registration_file = str(sys.argv[1])
    run_obstacle_detection(registration_file)




















