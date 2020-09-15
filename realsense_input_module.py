## 
#  @package realsense_input_module
#  Getting inputs from Intel Realsense SR300 and T265.
#  Contains stream enabling, getting frames and pose data.


##
# Import OpenCV, librealsense, numpy and logging

import pyrealsense2 as rs
import numpy as np
import cv2
import logging


## 
#  @brief Get configuration of the sensors. It is necessary for enable streams.
#  @return configuration of the sensors

def get_config():
    config = rs.config()
    return config


## 
#  @brief Get pipeline of the sensors. It is necessary for enable streams.
#  @return pipeline of the sensors

def get_pipeline():
    pipeline = rs.pipeline()
    return pipeline


## 
#  @brief Enable streams (fisheye1, fisheye2, pose) in case of T265
#  @param config is the config for T265

def enable_stream_t265(config):
    config.enable_stream(rs.stream.fisheye, 1) #RS2_FORMAT_Y8
    config.enable_stream(rs.stream.fisheye, 2)
    config.enable_stream(rs.stream.pose)


## 
#  @brief Enable streams (color, depth) in case of SR300
#  @param config is the config for SR300
#  @param rows is the rows for the resoluion, 640 is suggested
#  @param cols is the cols for the resoluion, 480 is suggested
#  @param framerate is the framerate, 30 is suggested

def enable_stream_sr300(config, rows, cols, framerate):
    config.enable_stream(rs.stream.depth, rows, cols, rs.format.z16, framerate)
    config.enable_stream(rs.stream.color, rows, cols, rs.format.bgr8, framerate)


## 
#  @brief Get frames of the sensors
#  @param pipeline is the pipeline
#  @return frames of the sensors

def get_frames(pipeline):
    frames = pipeline.wait_for_frames()
    return frames


## 
#  @brief Get fisheye frames
#  @param frames is the frame coming from the sensor
#  @param camera_num is the number of the fisheye camera (1 or 2)
#  @return the fisheye frame

def get_fisheye_frames(frames, camera_num):
    fisheye_frame = frames.get_fisheye_frame(camera_num)
    return fisheye_frame


## 
#  @brief Get pose frames
#  @param frames is the frame coming from the sensor
#  @return the pose frame

def get_pose_frames(frames):
    pose_frame = frames.get_pose_frame()
    return pose_frame


## 
#  @brief Get pose data 
#  @param pose_frame is the pose frame
#  @return the pose data

def get_pose_data(pose_frame):
    pose_data = pose_frame.get_pose_data()
    return pose_data


## 
#  @brief Get depth frames
#  @param frames is the frames coming from the sensor
#  @return the depth frame

def get_depth_frames(frames):
    depth_frame = frames.get_depth_frame()
    return depth_frame


## 
#  @brief Get color frames
#  @param frames is the frames coming from the sensor
#  @return the color frame

def get_color_frames(frames):
    color_frame = frames.get_color_frame()
    return color_frame


## 
#  @brief Convert frames to nparrays for later usage
#  @param image frames is the image frames coming from the sensor
#  @return images in nparray

def convert_img_to_nparray(image_frames):
    img_to_nparray = np.asanyarray(image_frames.get_data())
    return img_to_nparray


## 
#  @brief Create a colormap to visualize depth image
#  @param depth_img_nparray is the depth image in nparray
#  @param alpha_n is a scale for depth map (0.03 is suggested)
#  @return the depth colormap

def get_depth_colormap(depth_img_nparray, alpha_n):
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img_nparray, alpha=alpha_n), cv2.COLORMAP_JET)
    return depth_colormap


## 
#  @brief Visualize frames
#  @param window_name is the name of the window
#  @param images_nparray is the image to be visualized in nparray
#  @param wait_key is the key to kill the wondow

def show_imgs(window_name, images_nparray, wait_key):
     cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
     cv2.imshow(window_name, images_nparray)
     cv2.waitKey(wait_key)


## 
#  @brief Get the velocity from T265
#  @param pose_data is the pose data from T265
#  @return the velocity value

def get_velocity(pose_data):
     velocity = pose_data.velocity
     return velocity


## 
#  @brief Get the translation from T265
#  @param pose_data is the pose data from T265
#  @return the translation value

def get_translation(pose_data):
     translation = pose_data.translation
     return translation


## 
#  @brief Get the acceleration from T265
#  @param pose_data is the pose data from T265
#  @return the acceleration value

def get_acceleration(pose_data):
     acceleration = pose_data.acceleration
     return acceleration


## 
#  @brief Get the rotation from T265
#  @param pose_data is the pose data from T265
#  @return the rotation value

def get_rotation(pose_data):
     rotation = pose_data.rotation
     return rotation


## 
#  @brief Get the angular velocity from T265
#  @param pose_data is the pose data from T265
#  @return the angular velocity value

def get_angular_velocity(pose_data):
     angular_velocity = pose_data.angular_velocity
     return angular_velocity


## 
#  @brief Get the angular acceleration from T265
#  @param pose_data is the pose data from T265
#  @return the angular acceleration value

def get_angular_acceleration(pose_data):
     angular_acceleration = pose_data.angular_acceleration
     return angular_acceleration


## 
#  @brief Get the tracker confidence from T265
#  @param pose_data is the pose data from T265
#  @return the velocity value

def get_tracker_confidence(pose_data):
     tracker_confidence = pose_data.tracker_confidence
     return tracker_confidence


## 
#  @brief Get the mapper confidence from T265
#  @param pose_data is the pose data from T265
#  @return the mapper confidence value

def get_mapper_confidence(pose_data):
     mapper_confidence = pose_data.mapper_confidence
     return mapper_confidence




