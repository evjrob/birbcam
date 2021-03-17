FROM python:3.8-buster
RUN apt update
RUN apt-get install -y libgl1-mesa-glx netcdf-bin libnetcdf-dev \
libhdf5-dev libatlas-base-dev libhdf5-serial-dev
RUN pip install --upgrade pip
RUN pip install --upgrade cython
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
# Packages below are built from git
ADD requirements_extra.txt requirements_extra.txt
RUN pip install -r requirements_extra.txt
