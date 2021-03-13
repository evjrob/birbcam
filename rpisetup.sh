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
