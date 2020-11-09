## @package obstacle-detection
#  Test trained network
#
#  This module can be used to test the trained networks from the command line
#
#  For command line usage call python test.py -h for help

import numpy as np
import cv2 as cv
import network
import utils


## Run test
#
#  This function is used for running the test with the trained network
def run_test():

    # Make an instance of the network
    net = network.Network()

    # Capture video feed to make input for the model
    cap = utils.get_cap(0)
    prevgray = utils.get_grayImage(cap)
    ih,iw = prevgray.shape

    # Make video frame rectangular and resize it to 299x299 pixels
    prevgray=prevgray[:,int((iw-ih)/2):int((iw+ih)/2)]
    prevgray = cv.resize(prevgray, (299,299))

    while True:
        # Compute optical flow and construct the input for the network
        gray = utils.get_grayImage(cap)
        gray=gray[:,int((iw-ih)/2):int((iw+ih)/2)]
        gray = cv.resize(gray, (299,299))
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0)
        prevgray = gray
        conc_img = utils.form_network_input(gray, flow)

        print("Conc img shape: {}\n".format(conc_img.shape))

        # Make predictions with the network
        prediction,mask = net.predict(conc_img)

        # Visualization
        viz = cv.resize(np.reshape(mask,(30,30)), (299,299))
        viz_2 = cv.resize(np.reshape(prediction,(30,30)), (299,299))
        utils.visualize_flow(viz)
        utils.visualize_flow(viz_2, name='pred')
        utils.visualize_flow(gray, name='imgs')

if __name__ == "__main__":
    run_test()
