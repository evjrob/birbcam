from astral import LocationInfo
from astral.sun import sun
import base64
import cv2 as cv
import datetime as dt
from dateutil.tz import tzlocal
from io import BytesIO
import logging
import multiprocessing as mp
import numpy as np
import os
from PIL import Image
import pytz
import requests
from scipy.signal import medfilt2d
import shutil
import sqlite3
import time
import traceback
from fastai.vision.all import *
from flask_opencv_streamer.streamer import Streamer



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)

PROJECT_PATH = os.getenv("BIRBCAM_PATH", "../")
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_PATH, "data/"))
ROTATE_CAMERA = os.getenv("ROTATE_CAMERA", False)
CUSTOM_TIMEZONE = os.getenv("CUSTOM_TIMEZONE",  tzlocal())
BIRBCAM_LATITUDE = float(os.getenv("BIRBCAM_LATITUDE", "51.049999"))
BIRBCAM_LONGITUDE = float(os.getenv("BIRBCAM_LONGITUDE", "-114.066666"))
LOCATION_NAME = os.getenv("LOCATION", "Calgary")
REGION_NAME = os.getenv("REGION_NAME", "Canada")
MIN_CORRECT_CONF = os.getenv("MIN_CORRECT_CONF", 0.75)
MIN_UNREVIEWED_CONF = os.getenv("MIN_UNREVIEWED_CONF", 0.9)
ENABLE_STREAMER = os.getenv("ENABLE_STREAMER", False)

streamer_port = 3030
streamer_require_login = False

if ENABLE_STREAMER:
    streamer = Streamer(streamer_port, streamer_require_login)

# Database path
DB_PATH = os.getenv('DB_PATH', '../data/model_results.db')

# Model artifact path
MODEL_PATH = os.getenv('MODEL_PATH', '../models/birbcam_prod.pkl')

# Timezone for python datetime objects
if isinstance(CUSTOM_TIMEZONE, str):
    tz = pytz.timezone(CUSTOM_TIMEZONE)
else:
    tz = CUSTOM_TIMEZONE
utc_tz = pytz.timezone('UTC')

# Datetime format for printing and string representation
dt_fmt = '%Y-%m-%dT%H:%M:%S.%f'

# Where to save captured frames with changes
save_dir = os.path.join(DATA_DIR, 'imgs/')

# Make the save dir if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Create the Videoapture object for the webcam
capture = cv.VideoCapture()
capture.set(cv.CAP_PROP_FPS, 30)
CV_MASK_THRESH = 255   # Threhold for foreground mask: 255 indicates objects
CV_KERNEL_SIZE = 25    # nxn kernel size for 2d median filter on foreground mask

# Create the astral city location object
city = LocationInfo(LOCATION_NAME, REGION_NAME, tz, BIRBCAM_LATITUDE, BIRBCAM_LONGITUDE) 


def camera_loop(queue, stop_time):
    # Model params
    # https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html

    lr = 0.05           # learning rate for the background sub model
    burn_in = 30        # frames of burn in for the background sub model
    i = 0               # burn in iteration tracker
    # Create the background subtraction model
    backSub = cv.createBackgroundSubtractorMOG2()
    #backSub = cv.createBackgroundSubtractorKNN()
    # Open the webcam
    capture.open(0)
    current_time = dt.datetime.now(tz=tz)
    while current_time <= stop_time:
        current_time = dt.datetime.now(tz=tz)
        timestamp = current_time.strftime(dt_fmt)
        utc_timestamp = dt.datetime.now(tz=utc_tz).strftime(dt_fmt)[:-5]
        ret, frame = capture.read()
        if bool(ROTATE_CAMERA):
            frame = np.rot90(frame, k=-1)
        if ENABLE_STREAMER:
            streamer.update_frame(frame)
            if not streamer.is_streaming:
                streamer.start_streaming()
        if ret:
            fgMask = backSub.apply(frame, learningRate=lr)
            if i < burn_in:
                i += 1
                continue
            queue.put((frame, fgMask, timestamp, utc_timestamp))
    
    # Release the webcam        
    capture.release()


def prediction_cleanup():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    c = conn.cursor()
    # Remove the correctly predicted nones with high confidence
    c.execute('''SELECT * FROM results 
        WHERE (true_label = "none" 
        AND prediction = "none" 
        AND confidence >= ?);''', (MIN_CORRECT_CONF,))
    rows = c.fetchall()
    l = len(rows)
    logging.info(f'Removing {l} rows with true label of "none" and confidence >= {MIN_CORRECT_CONF}')
    for i, row in enumerate(rows):
        utc_datetime, datetime, file_name, prediction, confidence, label, _ = row
        if os.path.exists(f'{save_dir}{file_name}'):
            os.remove(f'{save_dir}{file_name}')
        c.execute('''DELETE FROM results WHERE utc_datetime = ?;''', (utc_datetime,))
    # Remove very high confidence none predictions more than a days old
    date_thresh = dt.datetime.now() - dt.timedelta(days=1)
    date_thresh = date_thresh.strftime(dt_fmt)
    c.execute('''SELECT * FROM results 
        WHERE true_label IS NULL 
        AND prediction= "none" 
        AND confidence > ?
        AND datetime <= ?;''', (MIN_UNREVIEWED_CONF, date_thresh,))
    rows = c.fetchall()
    l = len(rows)
    logging.info(f'Removing {l} rows with predicted label of "none" and confidence > {MIN_UNREVIEWED_CONF}')
    for i, row in enumerate(rows):
        utc_datetime, datetime, file_name, prediction, confidence, label, _ = row
        if os.path.exists(f'{save_dir}{file_name}'):
            os.remove(f'{save_dir}{file_name}')
        c.execute('''DELETE FROM results WHERE utc_datetime = ?;''', (utc_datetime,))
    conn.commit()
    conn.close()


def night_pause_loop(stop_time):
    prediction_cleanup()
    current_time = dt.datetime.now(tz=tz)
    if ENABLE_STREAMER:
        capture.open(0)
    while current_time < stop_time:
        if ENABLE_STREAMER:
            ret, frame = capture.read()
            streamer.update_frame(frame)
            if not streamer.is_streaming:
                streamer.start_streaming()
        else:
            # logarithmic sleeping to reduce iterations
            current_time = dt.datetime.now(tz=tz)
            time_diff_seconds = (stop_time - current_time).total_seconds()
            time.sleep((time_diff_seconds // 2) + 1)

    if ENABLE_STREAMER:
        capture.release()

def main_loop(queue):
    while True:
        try:
            # Figure out when to run the webcam based on dawn and dusk today
            current_time = dt.datetime.now(tz=tz)
            today_sun_times = sun(city.observer, date=dt.datetime.now(tz=tz), tzinfo=tz)
            today_start = today_sun_times['dawn']
            today_end = today_sun_times['dusk']
            
            # If current time is less than dawn today, then wait until then
            if current_time < today_start:
                logging.info(f'Delaying the cature of images until dawn at {today_start:{dt_fmt}}')
                night_pause_loop(today_start)
            
            # We can capture images, start the camera loop until sunset today
            elif current_time >= today_start and current_time <= today_end:
                logging.info(f'Capturing images until dusk at {today_end:{dt_fmt}}')
                camera_loop(queue, today_end)
            
            # Pause image capture until dawn tomorrow
            elif current_time > today_end:
                tomorrow_sun_times = sun(city.observer, date=dt.datetime.now(tz=tz) + dt.timedelta(days=1), tzinfo=tz)
                tomorrow_start = tomorrow_sun_times['dawn']
                logging.info(f'Delaying the cature of images until dawn at {tomorrow_start:{dt_fmt}}')
                night_pause_loop(tomorrow_start)
        except Exception as e:
            logging.error(traceback.format_exc())
            pass


            
def image_processor(queue, DB_PATH=DB_PATH, save_dir=save_dir, model_path=MODEL_PATH):
    x = None
    learn = load_learner(model_path)
    while True:
        try:
            # Threshold mask
            x = queue.get()
            frame, fgMask, timestamp, utc_timestamp = x
            fgMaskMedian = medfilt2d(fgMask, CV_KERNEL_SIZE)
            if (fgMaskMedian >= CV_MASK_THRESH).any():
                # Get the frame and timestamp for the image to be processed
                logging.debug(f'Processing image with timestamp {timestamp}')
                # Convert the OpenCV image from BGR to RGB for fastai
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # Get the predicted label and confidence
                pred = learn.predict(rgb_frame)
                labels = pred[0] 
                confidences = pred[2].tolist()
                if len(labels) == 0:
                    labels = ['none']
                    confidence = 1 - max(confidences)
                else:
                    confidence = min([c for c in confidences if c > 0.5])
                fname_label = '_'.join(labels)
                pred_label = ','.join(labels)
                # Save the image with time stamp and label
                filename = f'{timestamp}_{fname_label}.jpg'
                filepath = f'{save_dir}{timestamp}_{fname_label}.jpg'
                cv.imwrite(filepath, frame)
                # Write results to sqlite3 database
                conn = sqlite3.connect(DB_PATH, timeout=60)
                with conn:
                    conn.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?)", 
                        (utc_timestamp, timestamp, filename, pred_label, confidence, None, None))
                conn.close()
                logging.info(f'Processed image with timestamp {timestamp} and found label(s) {pred_label}')
        except Exception as e:
            logging.error(traceback.format_exc())
            pass


def main():
    # Use a multiprocessing queue to offload slow image processing 
    # to other processes/cores and keep the camera_loop from being 
    # blocked and missing frames.
    q = mp.Queue()
    p1 = mp.Process(target=main_loop, args=(q,))
    p2 = mp.Process(target=image_processor, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


if __name__ == '__main__':
    main()
