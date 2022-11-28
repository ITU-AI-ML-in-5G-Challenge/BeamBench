## SenseNet Docker file
## Maximilian Arnold

FROM nvidia/cuda:11.4.2-runtime-ubuntu18.04

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx python3 python3-pip zlib1g-dev libjpeg-dev git nano cmake curl
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Set python 3 as the default python
RUN update-alternatives --set python3 /usr/bin/python3.7

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall && \
    rm get-pip.py

# Install Torch and NNI and upgrade h5py
RUN pip3 install numpy
RUN pip3 install Pillow

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install matplotlib
RUN pip3 install utm
RUN pip3 install scikit-build
RUN pip3 install opencv-python
RUN pip3 install sklearn
RUN pip3 install tqdm
RUN pip3 install pandas
RUN pip3 install future
#RUN pip3 install feature
#RUN pip3 install annotations
RUN pip3 install open3d
RUN pip3 install h5py
