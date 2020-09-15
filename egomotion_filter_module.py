## 
#  @package egomotion_filter_module
#  Functions for optical flow egomotion filtering support.


##
# Import OpenCV, librealsense, numpy and logging

import pyrealsense2 as rs
import numpy as np
import cv2
import read_robot_states as robot_states
import numpy as np
import math
from timeit import default_timer as time


## 
#  @brief Convert RGB images to color images
#  @param color_image is the RGB image
#  @return gray image

def rgb_to_gray(color_image):
    gray = gray_prev = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    return gray


## 
#  @brief Get the number of rows of an image
#  @param image is the selected image
#  @return the number of rows of the image

def get_num_rows(image):
    num_rows = image.shape[0]
    return num_rows


## 
#  @brief Get the number of cols of an image
#  @param image is the selected image
#  @return the number of cols of the image

def get_num_cols(image):
    num_cols = image.shape[1]
    return num_cols


## 
#  @brief Visualize optical flow with arrows
#  @param img is the image
#  @param flow is the result of the optical flow
#  @param step is the pixel steps
#  @return the image with the arrows

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


## 
#  @brief Visualize optical flow with hsv
#  @param flow is the result of the optical flow
#  @return the hsv image

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


## 
#  @brief Visualize optical flow with image warping
#  @param img is the image
#  @param flow is the result of the optical flow
#  @return the warped image

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


## 
#  @brief Align the depth frame to color frame
#  @param frames is the frames what we want to align
#  @return the aligned depth frame

def align_depth_to_color(frames):
    align_to = rs.stream.color
    align = rs.align(align_to)
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    return aligned_depth_frame


## 
#  @brief Get ROI of an image
#  @param img is the src image
#  @param range_1 is the x coordinate start point
#  @param range_2 is the x coordinate end point
#  @param range_3 is the y coordinate start point
#  @param range_4 is the y coordinate end point
#  @return the roi

def get_roi(img, range_1, range_2, range_3, range_4):
    roi = img[range_1:range_2, range_3:range_4]
    return roi


## 
#  @brief Get ROI x of flow
#  @param flow is the result of optical flow
#  @param range_1 is the x coordinate start point
#  @param range_2 is the x coordinate end point
#  @param range_3 is the y coordinate start point
#  @param range_4 is the y coordinate end point
#  @return the x components of flow ROI

def get_flow_roi_x(flow, range_1, range_2, range_3, range_4):
    roi_flow_x = flow[range_1:range_2, range_3:range_4, 0]
    return roi_flow_x


## 
#  @brief Get ROI y of flow
#  @param flow is the result of optical flow
#  @param range_1 is the x coordinate start point
#  @param range_2 is the x coordinate end point
#  @param range_3 is the y coordinate start point
#  @param range_4 is the y coordinate end point
#  @return the y components of flow ROI

def get_flow_roi_y(flow, range_1, range_2, range_3, range_4):
    roi_flow_y = flow[range_1:range_2,range_3:range_4, 1]
    return roi_flow_y


## 
#  @brief Get the current and previous pixel locations.
#  @param img is the RGB image
#  @param flow is the optical flow matrix (2D)
#  @param step is the step number for index calculation
#  @return the current and previous pixel locations


def get_lines(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    return lines


## 
#  @brief Deprojection the pixels on the previous locations
#  @param aligned_depth_frame is the depth frame aligned to the color frame (or other two frames)
#  @param lines is the matrix which contains the previous and the current pixel locations
#  @param step is the step number for index calculation
#  @return the deprojected coordinates


def deproject_flow_prev(aligned_depth_frame, lines, step=16):
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    print(depth_intrin)
    h, w = depth_image.shape[:2]
    deprojected_coordinates = np.empty((h//step, w//step, 3))
    x_index_change = lines[:,:,0]
    y_index_change = lines[:,:,1]

    x_index_change = x_index_change.reshape(h//step, w//step, 2)
    y_index_change = y_index_change.reshape(h//step, w//step, 2)

    for i in range(h//step):
        for j in range(w//step):
            x_index_original = x_index_change[i,j,0]
            y_index_original = y_index_change[i,j,0]
            depth = aligned_depth_frame.get_distance(x_index_original, y_index_original)
            depth_point_in_camera_coords = np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, [x_index_original, y_index_original], depth))
            deprojected_coordinates[i, j] = depth_point_in_camera_coords
    return deprojected_coordinates


## 
#  @brief Deprojection the pixels on the current locations
#  @param aligned_depth_frame is the depth frame aligned to the color frame (or other two frames)
#  @param lines is the matrix which contains the previous and the current pixel locations
#  @param step is the step number for index calculation
#  @return the deprojected coordinates

def deproject_flow_new(aligned_depth_frame, lines, step=16):
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    h, w = depth_image.shape[:2]
    deprojected_coordinates = np.empty((h//step, w//step, 3))
    x_index_change = lines[:,:,0]
    y_index_change = lines[:,:,1]
    x_index_change = x_index_change.reshape(h//step, w//step, 2)
    y_index_change = y_index_change.reshape(h//step, w//step, 2)

    for i in range(h//step):
        for j in range(w//step):
            x_index_new = x_index_change[i,j,1]
            y_index_new = y_index_change[i,j,1]
            depth = aligned_depth_frame.get_distance(x_index_new, y_index_new)
            depth_point_in_camera_coords = np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, [x_index_new, y_index_new], depth))
            deprojected_coordinates[i, j] = depth_point_in_camera_coords
    return deprojected_coordinates


def velocity_from_point_clouds(deprojected_coordinates, T_cam_tcp, T_tcp_cam, robot_state_1, robot_state_2, robot_dt):
    h, w = deprojected_coordinates.shape[:2]
    velocities = np.empty((h, w, 3))
    T_2_1 = T_cam_tcp.dot(robot_state_2.T_tcp_base).dot(robot_state_1.T_base_tcp).dot(T_tcp_cam)
    for i in range(h):
        for j in range(w):
            homegenous_coords_1 = np.append(np.asmatrix(deprojected_coordinates[i,j,:]), np.matrix('1'), axis = 1).transpose()
            homegenous_coords_2 = T_2_1.dot(homegenous_coords_1)
            homegenous_velocities = np.asarray(((homegenous_coords_2 - homegenous_coords_1) / robot_dt).flatten())
            velocities[i,j,:] = homegenous_velocities[:,0:3]
    return velocities
    



## 
#  @brief Velocity comparison for tracking camera and depth camera
#  @param aligned_depth_frame is the depth frame aligned to the color frame (or other two frames)
#  @param diff_flow is the 3D optical flow
#  @param velocity is the velocity value coming from the tracking camera
#  @param velocity_robot_x is the velocity value x component coming from the robot
#  @param velocity_robot_y is the velocity value y component coming from the robot
#  @param velocity_robot_z is the velocity value z component coming from the robot
#  @param threshold is for the velocity difference calculations
#  @param step is the step number for index calculation
#  @return the egomotion filtered optical 3D optical flow


def velocity_comparison(aligned_depth_frame, diff_flow, velocities_from_egomotion, threshold, step=16):
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    h, w = depth_image.shape[:2]
    egomotion_filtered_flow = np.empty((h//step, w//step, 3))
    diff_flow_rel = np.empty((h//step, w//step, 3))
    for i in range(h//step):
        for j in range(w//step):
            #print("diff_flow:\t{}\t\tvelocity:\t{}".format(diff_flow[i,j,:],velocities_from_egomotion[i,j,:]))

            diff_flow_rel[i,j,0] = -(diff_flow[i,j,0] - velocities_from_egomotion[i,j,0])
            diff_flow_rel[i,j,1] = -(diff_flow[i,j,1] - velocities_from_egomotion[i,j,1])
            diff_flow_rel[i,j,2] = -(diff_flow[i,j,2] - velocities_from_egomotion[i,j,2])

            if (abs(diff_flow_rel[i,j,0]) > threshold or abs(diff_flow_rel[i,j,1]) > threshold or abs(diff_flow_rel[i,j,2]) > threshold):
                egomotion_filtered_flow[i,j] = diff_flow_rel[i,j]
  
            else:
                 egomotion_filtered_flow[i,j] = 0
    return egomotion_filtered_flow




## 
#  @brief Get the filtered flow in 2D for visualization
#  @param egomotion_filtered_flow is the filtered 3D optical flow
#  @param flow is the 2D optical flow
#  @return the filtered flow in 2D


def filtered_flow_2d(egomotion_filtered_flow, flow, step=16):
    h, w = flow.shape[:2]
    filtered_2d = np.zeros((h,w,2))
    for i in range(h//step):
        for j in range(w//step):
            if not(egomotion_filtered_flow[i,j,0] == 0 and
            egomotion_filtered_flow[i,j,1] == 0 and
            egomotion_filtered_flow[i,j,2] == 0):
            
                # Fill step grid with the current value
                for k in range(step):
                    for l in range(step):
                        filtered_2d[i*step + k,j*step + l,0] = \
                            flow[i*step + (step//2),j*step + (step//2),0]
                
                        filtered_2d[i*step + k,j*step + l,1] = \
                            flow[i*step + (step//2),j*step + (step//2),1]
    #print(filtered_2d)                
    return filtered_2d


## 
#  @brief Get the 3D optical flow
#  @param deproject_flow_new is the recent deprojected points
#  @param deproject_flow is the prior deprojected points
#  @return the 3D optical flow


def flow_3d(deproject_flow_new, deproject_flow, dt):
    h, w = deproject_flow_new.shape[:2]
    flow_3d = np.empty((h,w,3))
    for i in range(h):
        for j in range(w):
            if deproject_flow_new[i,j,2] == 0 or deproject_flow[i,j,2] == 0 :
                flow_3d[i,j] = 0
            else :
                flow_3d[i,j] = (deproject_flow_new[i,j] - deproject_flow[i,j]) / dt
    
    return flow_3d
    

## 
#  @brief Get the distances
#  @param aligned_depth_frame is the depth frame aligned to the color frame (or other two frames)
#  @param range_1 is the x and y coordinate start point
#  @param range_2 is the x and y coordinate end point
#  @return the depth array


def get_distance(aligned_depth_frame, range_1, range_2):
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_array_size_x = range_2 - range_1
    depth_array_size_y = range_2 - range_1
    depth_array = np.zeros((depth_array_size_x, depth_array_size_y))
    for i in range(range_1 , range_2):
        for j in range(range_1, range_2):
            depth = aligned_depth_frame.get_distance(i, j)
            depth_array[i - range_1 , j - range_2] = depth
    return depth_array


## 
#  @brief Deprojection the pixels
#  @param aligned_depth_frame is the depth frame aligned to the color frame (or other two frames)
#  @param range_1 is the x and y coordinate start point
#  @param range_2 is the x and y coordinate end point
#  @return the deprojected coordinates


#def deproject(aligned_depth_frame, range_1, range_2):
#    depth_image = np.asanyarray(aligned_depth_frame.get_data())
#    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
#    deprojected_coordinates = np.empty((range_2 - range_1, range_2 - range_1, 3))
#    for i in range(range_1, range_2):
#        for j in range(range_1, range_2):
#            depth = aligned_depth_frame.get_distance(i, j)
#            depth_point_in_camera_coords = np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, [i, j], depth))
#            deprojected_coordinates[i - range_1 , j - range_2] = depth_point_in_camera_coords
#    return deprojected_coordinates

def deproject(aligned_depth_frame, step=16):
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    h, w = depth_image.shape[:2]
    deprojected_coordinates = np.empty((h//step, w//step, 3))

    for i in range(h//step):
        for j in range(w//step):
            depth = aligned_depth_frame.get_distance(i,j)
            depth_point_in_camera_coords = np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, [i,j], depth))
            deprojected_coordinates[i, j] = depth_point_in_camera_coords
    return deprojected_coordinates



## 
#  @brief Get the extrinsics between two sensors
#  @param frame_1 is the frame from the first sensor
#  @param frame_2 is the frame from the second sensor
#  @return the extrinsics

def get_extrinsics(frame_1, frame_2):
    depth_to_color_extrin = frame_1.profile.get_extrinsics_to(frame_2.profile)
    return depth_to_color_extrin


## 
#  @brief Get points in an other coordinate system
#  @param extrinsics is the extrinsics between the two sensors
#  @param deprojected_coordinates is the 3D coordinates of the sensor
#  @return the point in the new coordinate system

def point_to_another_coord(extrinsics, deprojected_coordinates):
    new_point = rs.rs2_transform_point_to_point(extrinsics, deprojected_coordinates)
    return new_point



def calc_mean_velocity(egomotion_filtered_flow):
    velocity_mean_nonzero_elements = [0.0, 0.0, 0.0]
    h, w = egomotion_filtered_flow.shape[:2]
    velocity_nonzero_elements=[]
    for i in range(h):
         for j in range(w):
            if (egomotion_filtered_flow[i,j,0]!=0 and\
              egomotion_filtered_flow[i,j,1]!=0 and\
              egomotion_filtered_flow[i,j,2]!=0):
                velocity_nonzero_elements.append(egomotion_filtered_flow[i,j,:])

    velocity_mean_nonzero_elements = np.mean(velocity_nonzero_elements,0)
    velocity_std_nonzero_elements = np.std(velocity_nonzero_elements,0)
    return velocity_mean_nonzero_elements, velocity_std_nonzero_elements



def calc_mean_depth(egomotion_filtered_flow, deproject_flow_new):
    h, w = egomotion_filtered_flow.shape[:2]
    depth_z=[]
    for i in range(h):
        for j in range(w):
             depth_z.append(deproject_flow_new[i,j,2])

    depth_mean_z = np.mean(depth_z,0)
    depth_std_z = np.std(depth_z,0)
    return depth_mean_z, depth_std_z

def calc_mean_depth_mask(egomotion_filtered_flow, deproject_flow_new, ball_mask, step):
    h, w = egomotion_filtered_flow.shape[:2]
    depth_z=[]
    for i in range(h):
        for j in range(w):
            if (ball_mask[i*step + step//2,j*step + step//2] > 0):
                depth_z.append(deproject_flow_new[i,j,2])

    depth_mean_z = np.mean(depth_z,0)
    depth_std_z = np.std(depth_z,0)
    return depth_mean_z, depth_std_z


def show_pointcloud(v, deproject_flow_new_flat, three_d_flow_x):
    v.color_map('cool')
    v.clear()
    v.load(deproject_flow_new_flat)
    v.attributes(three_d_flow_x)
    v.color_map('jet',[-0.005, 0.005])
    v.set(point_size=0.001, lookat=[-0.03220592,  0.10527971,  0.20711438],phi=-1.00015545, theta=-2.75502944, r=0.36758572)



































