# Obstacle Detection

This repository contains the code for detecting moving obstacles, based on egomotion filtered optical flow, for the navigation of an autonomous vehicle.

## Installation procedure

After the `deploy` branch is checked out navigate inside the repository in the command line and run the install script by calling:

```bash
./scripts/INSTALL.sh
```

This will promt you for giving sudo password and then start the install. The script should install all the necessary software. This should take up to a few minutes because of downloading the model definitions of the neural network from git-lfs.

After the installation is complete, the pre-built packages of librealsense should be installed. For this follow the instructions [here](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md), like this:

 - Register the keyserver's public key:
    ```bash
    sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
    ```

 - Add the server to the list of repositories:

    Ubuntu 16 LTS:
    ```bash
    sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u
    ```
    Ubuntu 18 LTS:
    ```bash
    sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
    ```

 - Install the libraries:
 ```bash
 sudo apt-get install librealsense2-dkms
 sudo apt-get install librealsense2-utils
 ```

  - Optionally install the developer and debug packages:
  ```bash
  sudo apt-get install librealsense2-dev
  sudo apt-get install librealsense2-dbg
  ```

After these steps reboot your machine and disable secure boot in the system setup before booting Ubuntu again.

After the reboot (with secure boot disabled) run
```bash
realsense-viewer
```
in the command line. When the Realsense camera is connected to the machine its data should be available in Realsense-Viewer.

In order to test the scripts, first the python virtual environment should be activated, because all the necessary python packages are installed inside
the virtual environment.

To activate the virtual environment call:
```bash
source ../environments/obstacle_detection/bin/activate
```

After these verify that `obstacle_detection.py` runs without errors by calling:
```bash
python obstacle_detection.py
```
from the command line inside the repository.


## Usage

In order to run the scripts, first the python virtual environment should be activated, because all the necessary python packages are installed inside
the virtual environment.

To activate the virtual environment call:
```bash
source ../environments/obstacle_detection/bin/activate
```

### Obstacle Detection

The obstacle detection can be used by running
```bash
python obstacle_detection.py
```

The scipt also has a command line interface. See the help by calling
```bash
python obstacle_detection.py -h
```

With the command line interface, the obstacle detection can be run in two operating modes. When it is run like
```bash
python obstacle_detection.py
```
Only minimal information is printed in the console and only the results are sent on port **5555** in ZMQ.

If the script is called in debug mode with
```bash
python obstacle_detection.py -debug
```
then intermediate results are printed to the command line and the frames, the prediction of the network, the computed mask and blob and the original and corrected optical flows are also shown for debugging purpose.

With the `-filename` argument the name of the required config file can be chosen. This config file should contain the transformation between the robot's and the camera's coordinate frames.

### ZMQ Clients

With calling 
```bash
python zmq_client_sensorfusion.py
```
The data received from the sensor-fusion module is printed to the command line. This script can be used to check if the connection to the ZMQ publisher is live or not.

In order to test the if the publicatin of the results are successful and to see an example for the processing of the resulted messages use
```bash
python zmq_client_obstacle.py
```

### Cameras and data_collection

Use the `data_collection_realsense.py` module to check if the program can work with the Realsense camera or not:
```bash
python data_collection_realsense.py
```
This module also has a command line interface. Call its help to see the description:
```bash
python data_collection_realsense.py -h
```
If the script is called without arguments it can be used to check if the data from the Realsense camera can be received.

The script can be used to record data for further processing, if it is called like:
```bash
python data_collection_realsense.py -record
```
In this case the data from the camera will be recorded in a new folder called `data` in a .bag file. It also saves the incoming data from the sensor fusion in a .txt file in the root of the repository. The name of the files can be set by the -filename argument.

Calling the script with the -playback option enables to check the contents of the previously recorded .bag files. The file can be selected with the -filename argument.

The `data_collection_webcam.py` module is a fallback in-case the realsense camera is not present, or only the image data is needed. This script uses the default webcam of the machine instead of the Realsense camera and saves videos in the `data` folder.