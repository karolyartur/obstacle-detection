## @package obstacle-detection
#  Realsense Data Collection
#
#  This module can be used to save realsense data to a bag file and to load
#  the contents of a bag file for playback or data preparation

import os
import pyrealsense2 as rs
import numpy as np
import cv2
import realsense_input_module as ri
import argparse
import ur_tests_module as tests
import Msg
import zmq
import flatbuffers
import time


## Parse command line
#
#  This function parses the command line arguments
def parse_args():

    parser = argparse.ArgumentParser(
        description = 'View/record/prepare realsense data')
    parser.add_argument(
        '-minimal', dest='minimal',
        help='Boolean swith to switch visualization on and off. Default is: true (on)',
        default=True, action='store_false')
    parser.add_argument(
        '-record', dest='record',
        help='Boolean swith to switch recording to a bag file on and off. Default is: false (off)',
        default=False, action='store_true')
    parser.add_argument(
        '-playback', dest='playback',
        help='Boolean swith between using the realsense camera or a recorded file. Default is: false (off)',
        default=False, action='store_true')
    parser.add_argument(
        '-filename', dest='filename',
        help='Name of the bag file. Default is: data',
        default='data', type=str)

    args = parser.parse_args()
    return args

## Main function
#
#  This function is used for recording data with the Realsense camera
#  @param args Object for passing command line options to the function.
def main(args):

    with open('data.txt', 'w') as f:
        while True:

            if use_zmq_data:
                socket.recv()
                array_buffer = socket.recv()
                data_bytearray = bytearray(array_buffer)
                msg = Msg.Msg.GetRootAsMsg(array_buffer,0)
                msg_as_np = msg.ValueVectorAsNumpy()
            else:
                msg_as_np = np.array([0,0,0,0,0])
            msg_time = time.time()
            if args.record:
                f.write(np.array_str(msg_as_np)+','+str(msg_time-start_time)+'\n')

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
            if args.minimal:
                ri.show_imgs('SR300', images, 1)

if __name__ == "__main__":

    args = parse_args()

    if not os.path.exists('data'):
        os.makedirs('data')

    if args.playback:
        args.record = False

    # Get configs for the two streams
    config_1 = ri.get_config()

    # Get pipelines for the two streams
    pipeline_1 = ri.get_pipeline()

    # Enable SR300 stream
    ri.enable_stream_sr300(config_1, 640, 480, 30)

    if args.playback:
        profile_2 = tests.start_sr300_2(config_1, pipeline_1, 640, 480, 30, 'data/' + args.filename +'.bag')

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    use_zmq_data = True
    socket.connect("tcp://192.168.1.242:1111")
    socket.subscribe(b'd\x02\x00\x03')
    try:
        print(socket.recv())
        socket.recv()
        print('Received data from ZMQ')
        print("\nConnected to ZMQ\n")
    except:
        use_zmq_data = False
        print('\nUnable to get data from ZMQ!')

    if args.record:
        config_1.enable_record_to_file('data/' + args.filename + '.bag')

    start_time = time.time()
    # Start pipeline
    if not args.playback:
        pipeline_1.start(config_1)

    try:
        main(args)

    except KeyboardInterrupt:
        pass

    finally:
        # Stop streaming
        pipeline_1.stop()

