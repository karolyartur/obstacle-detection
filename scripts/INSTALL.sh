#!/bin/bash

curr_user=$(whoami)
#[ "$UID" -eq 0 ] || exec sudo bash "$0" "$@"0

# Install git
sudo apt-get update
sudo apt-get install -y git

# Install git lfs
sudo apt-get install -y software-properties-common
sudo curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git-lfs
git lfs install

# Pulling large file
echo "Pulling model file"
git lfs fetch
git lfs pull

# Set permissions
#sudo chown -R $curr_user .git/
echo "Repository configured"

# Install python and pip
sudo apt-get install -y python3.6
sudo apt-get install -y python3-pip
sudo apt-get install -y build-essential libssl-dev libffi-dev python-dev

# Install venv
sudo apt-get install -y python3-venv

# Create virtualenv and install pip packages
mkdir ../environments
cd ../environments
python3 -m venv obstacle_detection
#sudo chown -R $curr_user obstacle_detection
source obstacle_detection/bin/activate
cd ../obstacle-detection
pip install wheel
pip install -r requirements.txt

# Install tkinter for ZMQ
sudo apt-get install python3-tk

echo "Installation complete"
