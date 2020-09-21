## 
#  @package realsense_intrinsics_module
#  Functions for intrisics support


##
# Import OpenCV, librealsense, numpy and logging

import pyrealsense2 as rs
import numpy as np
import cv2


## 
#  @brief Get profile of frames
#  @param image_frames is the frames of the sensor images
#  @return profile of an image frame

def get_profile(image_frames):
    profile = image_frames.get_profile() 
    return profile


## 
#  @brief Get video stream profile of profile
#  @param profile is the profile of a sensor
#  @return the video stream profile

def get_video_stream_profile(profile):
    video_stream_profile = rs.video_stream_profile(profile)
    return video_stream_profile


## 
#  @brief Get the intrinsics of one sensor
#  @param video_stream_profile is the stream profile of the video
#  @return the intrinsics

def get_intrinsics(video_stream_profile):
    intrinsics = video_stream_profile.get_intrinsics()
    return intrinsics


## 
#  @brief Get the width of the frame
#  @param intrinsics is the intrinsic parameters of the sensor
#  @return the width

def get_width(intrinsics):
    width = intrinsics.width
    return width


## 
#  @brief Get the height of the frame
#  @param intrinsics is the intrinsic parameters of the sensor
#  @return the height

def get_height(intrinsics):
    height = intrinsics.height
    return height


## 
#  @brief Get the ppx of the frame
#  @param intrinsics is the intrinsic parameters of the sensor
#  @return the ppx

def get_ppx(intrinsics):
    ppx = intrinsics.ppx
    return ppx


## 
#  @brief Get the ppy of the frame
#  @param intrinsics is the intrinsic parameters of the sensor
#  @return the ppy	

def get_ppy(intrinsics):
    ppy = intrinsics.ppy
    return ppy


## 
#  @brief Get the fx of the frame
#  @param intrinsics is the intrinsic parameters of the sensor
#  @return the fx

def get_fx(intrinsics):
    fx = intrinsics.fx
    return fx


## 
#  @brief Get the fy of the frame
#  @param intrinsics is the intrinsic parameters of the sensor
#  @return the fy

def get_fy(intrinsics):
    fy = intrinsics.fy
    return fy

## 
#  @brief Get the coeffs of the frame
#  @param intrinsics is the intrinsic parameters of the sensor
#  @return the coeffs

def get_coeffs(intrinsics):
    coeffs = np.array(intrinsics.coeffs)
    coeffs = coeffs[0:4]
    return coeffs


## 
#  @brief Create the intrinsic matrix
#  @param intrinsics is the intrinsic parameters of the sensor
#  @return the K matrix

def get_K_matrix(intrinsics):
    ppx = get_ppx(intrinsics)
    ppy = get_ppy(intrinsics)
    fx = get_fx(intrinsics)
    fy = get_fy(intrinsics)
    K = np.array([[fx,0.0,ppx], [0.0,fy,ppy], [0.0,0.0,1.0]])
    return K


