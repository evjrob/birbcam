from astral import LocationInfo
from astral.sun import sun
import cv2 as cv
import datetime as dt
from fastai.vision.all import *
import logging
import multiprocessing as mp
import numpy as np
from PIL import Image
import pytz
from scipy.signal import medfilt2d
import sqlite3
import time


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
save_dir = 'imgs/'

# Create the Videoapture object for the webcam
capture = cv.VideoCapture()
capture.set(cv.CAP_PROP_FPS, 1)

# Database
conn = sqlite3.connect('data/model_results.db', timeout=15)
    
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


def night_pause_loop(stop_time):
    current_time = dt.datetime.now(tz=tz)
    while current_time < stop_time:
        current_time = dt.datetime.now(tz=tz)
        time_diff_seconds = (stop_time - current_time).total_seconds()
        # logarithmic sleeping to reduce iterations
        time.sleep((time_diff_seconds // 2) + 1)


def main_loop(queue):
    while True:
        # Figure out when to run the webcam based on dawn and dusk today
        current_time = dt.datetime.now(tz=tz)
        today_sun_times = sun(city.observer, date=dt.datetime.now(tz=tz), tzinfo=tz)
        today_dawn = today_sun_times['dawn']
        today_dusk = today_sun_times['dusk']
        
        # If current time is less than dawn today, then wait until then
        if current_time < today_dawn:
            logging.info(f'Delaying the cature of images until dawn at {today_dawn:{dt_fmt}}')
            night_pause_loop(today_dawn)
        
        # We can capture images, start the camera loop until dusk today
        elif current_time >= today_dawn and current_time <= today_dusk:
            logging.info(f'Capturing images until dusk at {today_dusk:{dt_fmt}}')
            camera_loop(queue, today_dusk)
        
        # Pause image capture until dawn tomorrow
        elif current_time > today_dusk:
            tomorrow_sun_times = sun(city.observer, date=dt.datetime.now(tz=tz) + dt.timedelta(days=1), tzinfo=tz)
            tomorrow_dawn = tomorrow_sun_times['dawn']
            logging.info(f'Delaying the cature of images until dawn at {tomorrow_dawn:{dt_fmt}}')
            night_pause_loop(tomorrow_dawn)

            
def image_processor(queue):
    learn = load_learner('models/birbcam_prod.pkl')
    x = None
    while True:
        x = queue.get()
        # Get the frame and timestamp for the image to be processed
        frame, timestamp, utc_timestamp = x
        logging.debug(f'Processing image with timestamp {timestamp}')
        # Convert the OpenCV image from BGR to RGB for fastai
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Get the predicted label and confidence
        pred = learn.predict(rgb_frame)
        label = pred[0]
        confidence = float(pred[2].numpy().max())
        # Save the image with time stamp and label
        filename = f'{save_dir}{timestamp}_{label}.jpg'
        cv.imwrite(filename, frame)
        # Write results to sqlite3 database
        with conn:
            conn.execute("INSERT INTO results VALUES (?,?,?,?,?,?)", 
                  (utc_timestamp, timestamp, filename, label, confidence, None))
        logging.info(f'Processed image with timestamp {timestamp} and found label {label}')


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