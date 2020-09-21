## @package obstacle_detection
#  Util module
#
#  This module contains utility functions for the package.

import cv2 as cv
import numpy as np

## Normalize optical flow
#
#  This function normalizes optical flow. The x and y channels are normalized separately.
#
#  Return value is np.float32, shape(h,w,c), the normalized optical flow.
#  @param flow Optical flow of np.float32, shape:(w,h,c), where w is width, h is height and c is channels
def normalize_flow(flow):

    import numpy as np

    flow_x_channel = flow[:,:,0]
    flow_y_channel = flow[:,:,1]

    flow_min_x = flow_x_channel.min()
    flow_max_x = flow_x_channel.max()
    
    flow_min_y = flow_y_channel.min()
    flow_max_y = flow_y_channel.max()
    
    flow_normalized_x = (flow_x_channel - flow_min_x)/(flow_max_x - flow_min_x) 

    flow_normalized_y = (flow_y_channel - flow_min_y)/(flow_max_y - flow_min_y) 

    flow_normalized_both_channel = np.array([flow_normalized_x, flow_normalized_y])
    flow_normalized_both_channel_transposed = np.transpose(flow_normalized_both_channel, (1,2,0))

    return flow_normalized_both_channel_transposed


## Return a list of files in a directory
#
#  This function returns the list of files, and the files only in a selected directory.
#
#  Returns a list containing the name of files in the directory.
#  @param dir A string containing the directory path
def onlyfiles(dir):

    import os

    return [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

## Video capture
#
#  This function captures frames with the webcam
#  @param video_filename Path to video file, or 0. If 0, the webcam is used as a source.
def get_cap(video_filename):
    cap = cv.VideoCapture(video_filename)
    global show_hsv
    show_hsv = False
    return cap

## Visualize flow
#
# This function can be used for visualizing optical flow
#  @param gray np.array, grayscale image
#  @param flow np.array, optical flow
#  @param name string, name for the pop-up window
def visualize_flow(gray, flow=None, name='mask'):
    cv.imshow(name, gray)
    global show_hsv
    show_hsv = False
    if show_hsv:
        cv.imshow('flow HSV', draw_hsv(flow))
    ch = cv.waitKey(5)
    if ch == 27:
        return False
    if ch == ord('1'):
        show_hsv = not show_hsv
        print('HSV flow visualization is', ['off', 'on'][show_hsv])
    return True

## Draw flow
#
#  Helper function to visualize optical flow
#  @param img np.array, image over which the optical flow will be drawn
#  @param flow np.array, optical flow
#  @param step integer, step size for visualiíation
def draw_flow(img, flow, step=6):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

## Convert image to grayscale
#
#  This function converts a videocapture to grayscale image
#  @param cap opencv videocapture
def get_grayImage(cap):
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

## Formulate input for network
#
#  This function creates the data structure that can be fed to the neural network
#  @param grayscale_image np.array, grayscale image (later)
#  @param optical_flow np.array, optical flow (between former and later frame)
def form_network_input(grayscale_image, optical_flow):
    flow_normalized = normalize_flow(optical_flow)
    grayscale_image = (grayscale_image-grayscale_image.min())/(grayscale_image.max()-grayscale_image.min())
    grayscale_image = np.reshape(grayscale_image,(299,299,1))
    conc_img = np.concatenate((flow_normalized,grayscale_image), axis=2)
    return conc_img
