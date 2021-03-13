apt update
apt install -y libsm6 libxrender1 libfontconfig1 libilmbase-dev \
libopenexr-dev libgstreamer1.0-dev libgtk2.0-dev netcdf-bin libnetcdf-dev \
libhdf5-dev libatlas-base-dev libhdf5-serial-dev gfortran libjpeg-dev zlib1g-dev
apt install -y python3-pip
pip3 install --upgrade pip
pip3 install --upgrade cython

# Install requirements for camera and webapp
pip3 install -r webapp/requirements.txt
pip3 install -r camera-app/requirements.txt

rm -rf /code/
mkdir /code/

# Build Torchvision ourselves
git clone --branch v0.9.0 https://github.com/pytorch/vision /code/torchvision
cd /code/torchvision
export BUILD_VERSION=0.9.0
sudo python3 setup.py install


# Build FastAI ourselves
git clone --branch v2.1.6 https://github.com/fastai/fastai/ /code/fastai
cd /code/fastai
export BUILD_VERSION=2.1.6
sudo python3 setup.py install