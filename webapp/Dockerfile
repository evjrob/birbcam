FROM tiangolo/uwsgi-nginx-flask:python3.8
RUN apt update
RUN apt install -y libgl1-mesa-glx
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt