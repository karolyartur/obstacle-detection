## 
#  @package egomotion_filter_module
#  Functions for optical flow egomotion filtering support.


##
# Import OpenCV, librealsense, numpy and logging

import pyrealsense2 as rs
import numpy as np
import cv2
import numpy as np
import math
from timeit import default_timer as time
import realsense_intrinsics_module as intr

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
    #print(depth_intrin)
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
            if (x_index_new < 0):
                x_index_new = 0
            if (y_index_new < 0):
                y_index_new = 0
            if (y_index_new >= h):
                y_index_new = h-1
            if (x_index_new >= w):
                x_index_new = w-1
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



def velocity_from_point_clouds_robot_frame(deprojected_coordinates_robot, \
                                        v_robot, omega_robot):
    if (v_robot.shape[0] != 3 or omega_robot.shape[0] != 3):
         raise Exception("v_robot and omega_robot is not passed as numpy column vector! v_robot.shape: {}, omega_robot.shape: {}".format(v_robot.shape, omega_robot.shape))

    h, w = deprojected_coordinates_robot.shape[:2]
    velocities = np.empty((h, w, 3))

    for i in range(h):
        for j in range(w):
            if np.isnan(deprojected_coordinates_robot[i,j,0]):
                velocities[i,j,:] = np.array([None, None, None])
            else:
                tangential_velocity = np.cross((-1.0 * omega_robot),\
                                                np.asmatrix(deprojected_coordinates_robot[i,j,:]).transpose(), axis=0)

                velocities[i,j,:] = np.asarray((tangential_velocity - v_robot).flatten())
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


def velocity_comparison(aligned_depth_frame, diff_flow, velocities_from_egomotion, threshold_lower, threshold_upper, step=16):
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    h, w = depth_image.shape[:2]
    egomotion_filtered_flow = np.empty((h//step, w//step, 3))
    diff_flow_rel = np.empty((h//step, w//step, 3))
    for i in range(h//step):
        for j in range(w//step):
            if np.isnan(diff_flow[i,j,0]) or np.isnan(velocities_from_egomotion[i,j,0]):
                diff_flow_rel[i,j,0] = 0
                diff_flow_rel[i,j,1] = 0
                diff_flow_rel[i,j,2] = 0
            else:
                diff_flow_rel[i,j,0] = -(diff_flow[i,j,0] - velocities_from_egomotion[i,j,0])
                diff_flow_rel[i,j,1] = -(diff_flow[i,j,1] - velocities_from_egomotion[i,j,1])
                diff_flow_rel[i,j,2] = -(diff_flow[i,j,2] - velocities_from_egomotion[i,j,2])

            if ((abs(diff_flow_rel[i,j,0]) > threshold_lower or abs(diff_flow_rel[i,j,1]) > threshold_lower or abs(diff_flow_rel[i,j,2]) > threshold_lower) and \
                    (abs(diff_flow_rel[i,j,2]) < threshold_upper)):
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
                flow_3d[i,j] = np.array([None, None, None])
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
            if (egomotion_filtered_flow[i,j,0]!=0 and\
              egomotion_filtered_flow[i,j,1]!=0 and\
              egomotion_filtered_flow[i,j,2]!=0):
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



def transform_velocites(diff_flow, T):

    h, w = diff_flow.shape[:2]
    velocities_transformed = np.empty((h, w, 3))

    for i in range(h):
        for j in range(w):
            if np.isnan(diff_flow[i,j,0]):
                velocities_transformed[i,j,:] = np.array([None,None,None])
            else:
                homegenous_velocities = np.append(np.asmatrix(diff_flow[i,j,:]), \
                                                np.matrix('0'), axis = 1).transpose()
                homegenous_velocities_transformed = T.dot(homegenous_velocities)
                velocities_transformed[i,j,:] = np.asarray((homegenous_velocities_transformed[0:3]).flatten())
    return velocities_transformed

def transform_points(deprojected_coordinates, T):
    h, w = deprojected_coordinates.shape[:2]
    points_transformed = np.empty((h, w, 3))

    for i in range(h):
        for j in range(w):
            if (np.max(np.abs(deprojected_coordinates[i,j,:])) > 0.000001):
                homegenous_points = np.append(np.asmatrix(deprojected_coordinates[i,j,:]), \
                                            np.matrix('1'), axis = 1).transpose()
                homegenous_points_transformed = T.dot(homegenous_points)
                points_transformed[i,j,:] = np.asarray((homegenous_points_transformed[0:3]).flatten())
            else:
                points_transformed[i,j,:] = np.array([None,None,None])

    return points_transformed


def get_3D_bounding_box(deprojected_coordinates_robot, mask):

    bb = {'x1': math.inf, 'y1': math.inf, 'z1': math.inf,\
          'x2': -math.inf, 'y2': -math.inf, 'z2': -math.inf}

    h, w = deprojected_coordinates_robot.shape[:2]
    for i in range(h):
        for j in range(w):
            #print(mask[i,j])
            if (mask[i,j] > 0 and deprojected_coordinates_robot[i,j,2] != 0):
                if (deprojected_coordinates_robot[i,j,0] < bb.get('x1')):
                    bb['x1'] = deprojected_coordinates_robot[i,j,0]
                    #print("x1:\t{}".format(deprojected_coordinates_robot[i,j,0]))
                if (deprojected_coordinates_robot[i,j,1] < bb.get('y1')):
                    bb['y1'] = deprojected_coordinates_robot[i,j,1]
                    #print("y1:\t{}".format(deprojected_coordinates_robot[i,j,1]))
                if (deprojected_coordinates_robot[i,j,2] < bb.get('z1')):
                    bb['z1'] = deprojected_coordinates_robot[i,j,2]
                    #print("z1:\t{}".format(deprojected_coordinates_robot[i,j,2]))
                if (deprojected_coordinates_robot[i,j,0] > bb.get('x2')):
                    bb['x2'] = deprojected_coordinates_robot[i,j,0]
                    #print("x2:\t{}".format(deprojected_coordinates_robot[i,j,0]))
                if (deprojected_coordinates_robot[i,j,1] > bb.get('y2')):
                    bb['y2'] = deprojected_coordinates_robot[i,j,1]
                    #print("y2:\t{}".format(deprojected_coordinates_robot[i,j,1]))
                if (deprojected_coordinates_robot[i,j,2] > bb.get('z2')):
                    bb['z2'] = deprojected_coordinates_robot[i,j,2]
                    #print("z2:\t{}".format(deprojected_coordinates_robot[i,j,2]))

    return bb


def find_largest_blob(mask):


    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
    else:
        print("No blob found")
        return np.zeros(mask.shape, np.uint8)

    # Output
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    out = cv2.bitwise_and(mask, out)
    return out


def avg_coords(deprojected_coordinates_robot, mask):
    h, w = deprojected_coordinates_robot.shape[:2]
    mask_deprojected_x=[]
    mask_deprojected_y=[]
    mask_deprojected_z=[]
    
    for i in range(h):
        for j in range(w):
            if (mask[i,j] > 0) and not np.isnan(deprojected_coordinates_robot[i,j,0]):
                mask_deprojected_x.append(deprojected_coordinates_robot[i,j,0])
                mask_deprojected_y.append(deprojected_coordinates_robot[i,j,1])
                mask_deprojected_z.append(deprojected_coordinates_robot[i,j,2])
                
    mean_x = np.mean(mask_deprojected_x)
    mean_y = np.mean(mask_deprojected_y)
    mean_z = np.mean(mask_deprojected_z)
    
    blob_mean_coords = np.array([mean_x, mean_y, mean_z])
    return blob_mean_coords


def max_blob_width(deprojected_coordinates_robot, mask, aligned_depth_frame, blob_mean_coords, step = 16):
    h, w = deprojected_coordinates_robot.shape[:2]
    rows_nonzero = np.count_nonzero(mask, axis=0)
    max_row_nonzero = np.max(rows_nonzero)
    max_row_indices = [i for i, j in enumerate(rows_nonzero) if j == max_row_nonzero]
    
    deprojected_coordinates_robot_blob_width = deprojected_coordinates_robot[max_row_indices[0],:,:]
    blob_width_coords_x = []
    blob_width_coords_y = []
    blob_width_coords_z = []
    width_x = 0.0
    #print(len(deprojected_coordinates_robot_blob_width))
    #for i in range(len(deprojected_coordinates_robot_blob_width)):
    for j in range(w):
        if (mask[max_row_indices[0],j] > 0 and deprojected_coordinates_robot[max_row_indices[0],j,2] != 0):
            blob_width_coords_x.append(deprojected_coordinates_robot[max_row_indices[0],j,0])
            blob_width_coords_y.append(deprojected_coordinates_robot[max_row_indices[0],j,1])
            blob_width_coords_z.append(deprojected_coordinates_robot[max_row_indices[0],j,2])
    if (blob_width_coords_x):
        width_x = abs(blob_width_coords_x[-1] - blob_width_coords_x[0])
        
    object_size_in_image = max_row_nonzero * step
    object_distance_from_camera = blob_mean_coords[2]
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    fx = intr.get_fx(depth_intrin)
    # object size in image = Object size * focal length / object distance from camera
    
    object_estimated_width = (object_size_in_image * object_distance_from_camera) / fx
    
    return object_estimated_width