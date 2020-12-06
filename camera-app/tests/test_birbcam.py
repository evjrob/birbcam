import cv2 as cv
from datetime import datetime as dt
import multiprocessing as mp
import pytest
import pytz
import sqlite3
import time

from birbcam import image_processor
from birbcam import dt_fmt, tz, utc_tz


def test_image_processor():
    test_db = 'tests/test_assets/test.db'
    test_save_dir = 'tests/test_temp/'
    test_img = 'tests/test_assets/test_img.jpg'
    current_time = dt.now(tz=tz)
    timestamp = current_time.strftime(dt_fmt)
    frame = cv.imread(test_img)
    utc_timestamp = dt.now(tz=utc_tz).strftime(dt_fmt)
    
    q = mp.Queue()
    p = mp.Process(target=image_processor, args=(q,test_db,test_save_dir))
    p.start()
    x = (frame, timestamp, utc_timestamp)
    q.put(x)
    p.join(5)

    conn = sqlite3.connect(test_db, timeout=60)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM results WHERE utc_datetime=?;", (utc_timestamp,))
    row = c.fetchone()
    count = row[0]
    c = conn.cursor()
    c.execute("DELETE FROM results;")
    conn.commit()
    conn.close()

    assert count == 1, "Did not find the expected row in the database!"

    q.close()
    p.terminate()
    return