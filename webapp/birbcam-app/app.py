import base64
import cv2 as cv
import numpy as np
import sqlite3

from flask import Flask, redirect, render_template, request

# Image directory
img_dir = '../../imgs/'

# Create flask app
app = Flask(__name__)

# Route for main visualization page
@app.route('/')
def main_visualization_page():
    start_date = '2020-11-25'
    end_date = '2020-12-31'
    things = ['1', '2', '3']

    return render_template('visualization.html', things=things)

# Route for model evaluation page
@app.route('/eval')
def model_evaluation_page():
    # Fetch a single un-reviewed image from the database
    conn = sqlite3.connect('../../data/model_results.db', timeout=15)
    query = '''SELECT utc_datetime, file_name, prediction 
               FROM results 
               WHERE true_label IS NULL 
               ORDER BY utc_datetime
               LIMIT 1;'''
    c = conn.cursor()
    c.execute(query)
    row = c.fetchone()
    c.execute('''SELECT AVG(CASE WHEN true_label IS NULL THEN 0 ELSE 1 END) FROM results;''')
    progress = c.fetchone()[0] * 100
    precision = '.2f'
    progress = f'{progress:{precision}}%'
    conn.close()

    if row is not None:
        show_eval = True
        utc_key = row[0]
        img_fn = row[1]
        label = row[2]

        # Read the image and histogram nomalize it
        img = cv.imread(f'../../imgs/{img_fn}')
        if img is None:
            print(utc_key)
            print(img_fn)
        img_y_cr_cb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(img_y_cr_cb)

        # Applying equalize Hist operation on Y channel.
        y_eq = cv.equalizeHist(y)

        img_y_cr_cb_eq = cv.merge((y_eq, cr, cb))
        equ = cv.cvtColor(img_y_cr_cb_eq, cv.COLOR_YCR_CB2BGR)
        res = np.hstack((img,equ)) #stacking images side-by-side

        # img io
        is_success, im_buf_arr = cv.imencode(".jpg", res)
        png_img_bytes = im_buf_arr.tobytes()
        img_b64 = 'data:image/png;base64,' + base64.b64encode(png_img_bytes).decode('utf8')
    else:
        show_eval = False
        img_fn = None
        img_b64 = None
        label = None
        utc_key = None
    return render_template('evaluate.html', progress=progress, show_eval=show_eval, filename=img_fn, 
                           img=img_b64, label=label, utc_key=utc_key)

# Route for model evaluation page
@app.route('/api/model_eval_submit', methods=['POST'])
def model_evaluation_api():
    utc_key = str(request.form['utc_key'])
    label = str(request.form['label'])
    print(f'key: {utc_key}, label: {label}')
    # Update the row in the database with the selected true label
    conn = sqlite3.connect('../../data/model_results.db', timeout=15)
    c = conn.cursor()
    c.execute("UPDATE results SET true_label=? WHERE utc_datetime=?;", (label, utc_key))
    conn.commit()
    conn.close()
    return redirect("/eval")

# Route for model evaluation page
@app.route('/api/data', methods=['POST'])
def get_data():
    start_date = str(request.form['start_date'])
    end_date = str(request.form['end_date'])
    print(f'start date: {start_date}, end_date: {end_date}')
    # Update the row in the database with the selected true label
    # conn = sqlite3.connect('../data/model_results.db', timeout=15)
    # c = conn.cursor()
    # c.execute("UPDATE results SET true_label=? WHERE utc_datetime=?;", (label, utc_key))
    # conn.commit()
    # conn.close()
    return redirect("/")
