import pyrealsense2 as rs
import numpy as np
import cv2
import realsense_input_module as ri
import egomotion_filter_module as emf
import matplotlib.pyplot as plt
import read_robot_states
import time
import sys
import pptk
import read_robot_states




def start_sr300(config, pipeline, width_sr300, height_sr300, framerate_sr300, filename):
    # Get the camera data from a bag file
    rs.config.enable_device_from_file(config, str(sys.argv[1]) + "/" + filename)

    # Enable stream
    config.enable_stream(rs.stream.depth, width_sr300, height_sr300, rs.format.z16, framerate_sr300)
    config.enable_stream(rs.stream.color, width_sr300, height_sr300, rs.format.bgr8, framerate_sr300)

    # Start pipeline
    profile_2=pipeline.start(config)

    # Playback from bag
    playback_2 = profile_2.get_device().as_playback()
    playback_2.set_real_time(False)

    return profile_2

def start_t265(config, pipeline, filename):

    # Get the camera data from a bag file
    rs.config.enable_device_from_file(config, str(sys.argv[1]) + "/" + filename)

    # Enable stream
    config.enable_stream(rs.stream.pose)

    # Start pipeline
    profile_1 = pipeline.start(config)

    # Playback from bag
    playback_1 = profile_1.get_device().as_playback()
    playback_1.set_real_time(False)

    return profile_1


def velocity_ball_to_camera_frame(velcity_ball, T_camera_tcp, T_base_base_ball, robot_state):
    homegenous_velocity_ball = np.append(np.asmatrix(velcity_ball), np.matrix('0'), axis = 1).transpose()
    homogenous_velocity_ball_cf = np.asarray(\
        T_camera_tcp.dot(robot_state.T_tcp_base).dot( \
        T_base_base_ball).dot(homegenous_velocity_ball))
    velocity_cf= homogenous_velocity_ball_cf[0:3,:].flatten()
    return velocity_cf

def segment_ball(color_image, gray, minRadius, maxRadius, radiusOffset):
    ball_mask = np.zeros(gray.shape, dtype=np.uint8)
    ball_detection = color_image.copy()

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
            dp=1, minDist=100, param1=100, param2=30,
            minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(ball_detection, (x, y), r+radiusOffset, (0, 255, 0), 2)
                cv2.rectangle(ball_detection, (x - 5, y - 5), (x + 5, y + 5),
                    (0, 128, 255), -1)
                cv2.circle(ball_mask, (x, y), r+radiusOffset, 255, -1)
    return ball_mask, ball_detection

def sync_robot_state(robot_states, start_time, i_prev, current_time, dt):
    try:
        i = i_prev + 1
        time_robot = robot_states[i_prev].timestamp - start_time
        while ((current_time -
                (robot_states[i].timestamp - start_time)) >= dt):
            i += 1
            time_robot = robot_states[i].timestamp - start_time
        return i, time_robot
    except IndexError:
        print("Not enogh robot states!\n")
        raise Exception("Not enogh robot states!")

def sync_imu(ri, pipeline_1, frames_t265, start_time, current_time, dt):
    time_imu = 0.0
    while ((current_time -
            (ri.get_pose_frames(frames_t265).get_timestamp() - start_time)) >= dt):
        frames_t265 = ri.get_frames(pipeline_1)
        time_imu = ri.get_pose_frames(frames_t265).get_timestamp() - start_time

    return frames_t265, time_imu

































