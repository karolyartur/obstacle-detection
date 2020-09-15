import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import realsense_input_module as ri

# Get configs for the two streams
config_1 = ri.get_config()

# Get pipelines for the two streams
pipeline_1 = ri.get_pipeline()

# Enable SR300 stream
ri.enable_stream_sr300(config_1, 640, 480, 30)

# Start pipeline
pipeline_1.start(config_1)


try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames_sr300 = ri.get_frames(pipeline_1)
        depth_frame = ri.get_depth_frames(frames_sr300)
        color_frame = ri.get_color_frames(frames_sr300)
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = ri.convert_img_to_nparray(depth_frame)
        color_image = ri.convert_img_to_nparray(color_frame)


        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = ri.get_depth_colormap(depth_image, 0.03)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        ri.show_imgs('SR300', images, 1)

        # Get pose data

finally:

    # Stop streaming
    pipeline_1.stop()

