from astral import LocationInfo
from astral.sun import sun
import base64
import cv2 as cv
import datetime as dt
from fastai.vision.all import *
from io import BytesIO
import logging
import multiprocessing as mp
import numpy as np
import os
from PIL import Image
import pytz
import requests
from scipy.signal import medfilt2d
from secrets import ENDPOINT
import shutil
import sqlite3
import time
import traceback


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)

# Timezone for python datetime objects
tz = pytz.timezone('America/Edmonton')
utc_tz = pytz.timezone('UTC')

# Datetime format for printing and string representation
dt_fmt = '%Y-%m-%dT%H:%M:%S'

# Where to save captured frames with changes
save_dir = '../imgs/'

# Create the Videoapture object for the webcam
capture = cv.VideoCapture()
capture.set(cv.CAP_PROP_FPS, 1)

# Database path
db_path = '../data/model_results.db'

# Model artifact path
model_path = '../models/birbcam_prod.pkl'
    
# Create the astral city location object
city = LocationInfo("Calgary", "Canada", "America/Edmonton", 51.049999, -114.066666) 


def camera_loop(queue, stop_time):
    # Model params
    # https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html
    mask_thresh = 255   # Threhold for foreground mask: 255 indicates objects
    kernel_size = 25    # nxn kernel size for 2d median filter on foreground mask
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
        utc_timestamp = dt.datetime.now(tz=utc_tz).strftime(dt_fmt)
        ret, frame = capture.read()
        if ret:
            frame = np.rot90(frame, k=-1)
            fgMask = backSub.apply(frame, learningRate=lr)
            if i < burn_in:
                i += 1
                continue

            # Threshold mask
            fgMaskMedian = medfilt2d(fgMask, kernel_size)
            if (fgMaskMedian >= mask_thresh).any():
                # Put the frame and corresponding timestamp into the 
                # queue to be processed by the fastai models
                queue.put((frame, timestamp, utc_timestamp))
                logging.info(f'Passed image with timestamp {timestamp} for processing')
    
    # Release the webcam        
    capture.release()


def prediction_cleanup():
    conn = sqlite3.connect(db_path, timeout=60)
    c = conn.cursor()
    # Remove the correctly predicted nones with high confidence
    c.execute('''SELECT * FROM results 
        WHERE (true_label = "none" 
        AND prediction = "none" 
        AND confidence >= 0.75);''')
    rows = c.fetchall()
    l = len(rows)
    logging.info(f'Removing {l} rows with true label of "none" and confidence >= 0.75')
    for i, row in enumerate(rows):
        utc_datetime, datetime, file_name, prediction, confidence, label, _ = row
        if os.path.exists(f'{save_dir}{file_name}'):
            os.remove(f'{save_dir}{file_name}')
        c.execute('''DELETE FROM results WHERE utc_datetime = ?;''', (utc_datetime,))
    # Remove very high confidence none predictions more than two days old
    date_thresh = dt.datetime.now() - dt.timedelta(days=2)
    date_thresh = date_thresh.strftime(dt_fmt)
    c.execute('''SELECT * FROM results 
        WHERE true_label IS NULL 
        AND prediction= "none" 
        AND confidence > 0.95 
        AND datetime <= ?;''', (date_thresh,))
    rows = c.fetchall()
    l = len(rows)
    logging.info(f'Removing {l} rows with predicted label of "none" and confidence > 0.95')
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
    while current_time < stop_time:
        current_time = dt.datetime.now(tz=tz)
        time_diff_seconds = (stop_time - current_time).total_seconds()
        # logarithmic sleeping to reduce iterations
        time.sleep((time_diff_seconds // 2) + 1)


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


            
def image_processor(queue, db_path=db_path, save_dir=save_dir, model_path=model_path):
    # learn = load_learner(model_path)
    x = None
    while True:
        try:
            x = queue.get()
            # Get the frame and timestamp for the image to be processed
            frame, timestamp, utc_timestamp = x
            logging.debug(f'Processing image with timestamp {timestamp}')
            # Convert the OpenCV image from BGR to RGB for fastai
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Get the predicted label and confidence
            pil_img = Image.fromarray(rgb_frame)
            buf = BytesIO()
            pil_img.save(buf, format="JPEG")
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            r = requests.post(ENDPOINT, json={"b64_img":b64_img})
            results = r.json()
            labels = results['labels']
            confidences = [c[1] for c in results['confidence']]
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
            conn = sqlite3.connect(db_path, timeout=60)
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