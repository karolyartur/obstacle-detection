## @package obstacle_detection
#  Record videos for creating labeled data for the training of the network
#
#  This module can be used to record video sequences for creating the training data from
#  It can be used by importing the package, or this module in a python module, or from the command line.
#
#  For command line usage call python create_dataset.py -h for help


import os
import cv2
import sys
import numpy as np

## Parse command line
#
#  This function parses the command line arguments
def parse_args():

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Record Video Sequences')
    parser.add_argument(
        '-tdir', dest='tdir',
        help='Directory relative to cwd to save videos to. Default is: vids',
        default='vids', type=str)

    args = parser.parse_args()
    return args


## Record videos
#
#  This function is used for recording videos
#  @param args Object for passing the options. Use the VID class of ginop.utils module.
def record_videos(args):

    vidcap = cv2.VideoCapture(0)
    count = 0
    frame_num = 0

    success,image = vidcap.read()
    ih,iw,ic = image.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('data/' + args.tdir + '/' + str(count).zfill(5) + '.avi', fourcc, 20.0, (iw,ih))
    print('data/' + args.tdir + '/' + str(count).zfill(5) + '.avi')

    full_video_writer = cv2.VideoWriter(os.path.join('data',args.tdir,'full','full.avi'), fourcc, 20.0, (iw,ih))

    try:
        while(True):
            success,image = vidcap.read()
            ih,iw,ic = image.shape

            if success:
                frame_num += 1
                out.write(image)
                full_video_writer.write(image)
                if frame_num == 100:
                    out.release()
                    count += 1
                    frame_num = 0
                    out = cv2.VideoWriter('data/' + args.tdir + '/' + str(count).zfill(5) + '.avi', fourcc, 20.0, (iw,ih))        

    except KeyboardInterrupt as k:
        full_video_writer.release()
        out.release()
        vidcap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.path.insert(0,'..')
    args = parse_args()

    if not os.path.exists(os.path.join('data',args.tdir)):
        os.makedirs('data/' + args.tdir)
    if not os.path.exists(os.path.join('data',args.tdir,'full')):
        os.makedirs(os.path.join('data',args.tdir,'full'))
    
    record_videos(args)
