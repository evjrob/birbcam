from astral import LocationInfo
from astral.sun import sun
import cv2 as cv
import datetime as dt
import logging
import numpy as np
from scipy.signal import medfilt2d
import time
import pytz


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

# Where to save captured frames with changes
save_dir = 'auto_imgs/'

# Create the Videoapture object for the webcam
capture = cv.VideoCapture()
capture.set(cv.CAP_PROP_FPS, 1)
    
# Create the astral city location object
city = LocationInfo("Calgary", "Canada", "America/Edmonton", 51.049999, -114.066666) 


def camera_loop(stop_time):
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
        timestamp = current_time.strftime('%Y-%m-%d_%H:%M:%S')
        _, frame = capture.read()
        frame = np.rot90(frame, k=-1)
        fgMask = backSub.apply(frame, learningRate=lr)
        if i < burn_in:
            i += 1
            continue
        
        # Threshold mask
        fgMaskMedian = medfilt2d(fgMask, kernel_size)
        if (fgMaskMedian >= mask_thresh).any():
            filename = f'{save_dir}{timestamp}.jpg'
            cv.imwrite(filename, frame)
            logging.info('Wrote file ' + filename) 
    
    # Release the webcam        
    capture.release()


def night_pause_loop(stop_time):
    current_time = dt.datetime.now(tz=tz)
    while current_time < stop_time:
        current_time = dt.datetime.now(tz=tz)
        time_diff_seconds = (stop_time - current_time).total_seconds()
        # logarithmic sleeping to reduce iterations
        time.sleep((time_diff_seconds // 2) + 1)


def main_loop():
    while True:
        # Figure out when to run the webcam based on dawn and dusk today
        current_time = dt.datetime.now(tz=tz)
        today_sun_times = sun(city.observer, date=dt.datetime.now(tz=tz), tzinfo=tz)
        today_dawn = today_sun_times['dawn']
        today_dusk = today_sun_times['dusk']
        
        # If current time is less than dawn today, then wait until then
        if current_time < today_dawn:
            logging.info(f'Delaying the cature of images until dawn at {today_dawn}')
            night_pause_loop(today_dawn)
        
        # We can capture images, start the camera loop until dusk today
        elif current_time >= today_dawn and current_time <= today_dusk:
            logging.info(f'Capturing images until dusk at {today_dusk}')
            camera_loop(today_dusk)
        
        # Pause image capture until dawn tomorrow
        elif current_time > today_dusk:
            tomorrow_sun_times = sun(city.observer, date=dt.datetime.now(tz=tz) + dt.timedelta(days=1), tzinfo=tz)
            tomorrow_dawn = tomorrow_sun_times['dawn']
            logging.info(f'Delaying the cature of images until dawn at {tomorrow_dawn}')
            night_pause_loop(tomorrow_dawn)


if __name__ == '__main__':
    main_loop()