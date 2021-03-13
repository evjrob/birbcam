FROM fastdotai/fastai:2.1.6
RUN apt install -y libgl1-mesa-glx
RUN pip install --upgrade pip
RUN pip install --upgrade cython
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
