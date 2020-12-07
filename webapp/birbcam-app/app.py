import base64
import cv2 as cv
import io
import json
import numpy as np
from PIL import Image
import sqlite3

from flask import Flask, redirect, render_template, request, jsonify, send_file

# Image directory
img_dir = '../../imgs/'

# Create flask app
app = Flask(__name__)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True


def histogram_equalize(img):
    img_y_cr_cb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(img_y_cr_cb)
    # Applying equalize Hist operation on Y channel.
    y_eq = cv.equalizeHist(y)
    img_y_cr_cb_eq = cv.merge((y_eq, cr, cb))
    equ = cv.cvtColor(img_y_cr_cb_eq, cv.COLOR_YCR_CB2RGB)
    return equ

def brighten_image(img, minimum_brightness=0.66):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cols, rows = gray_img.shape
    brightness = np.sum(gray_img) / (255 * cols * rows)
    ratio = brightness / minimum_brightness
    if ratio < 1:
        bright = cv.convertScaleAbs(img, alpha = 1 / ratio, beta = 0)
    else:
        bright = img
    bright = cv.cvtColor(bright, cv.COLOR_BGR2RGB)
    return bright

# Route for main visualization page
@app.route('/')
def main_visualization_page():
    return render_template('index.html')

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
        
        adj = brighten_image(img)
        res = np.hstack((img,adj)) #stacking images side-by-side

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
    labels = ['chickadee', 'magpie', 'sparrow', 'squirrel']
    json_data = request.get_json(force=True)
    start_date = json_data['start_date']
    end_date = json_data['end_date']
    print(f'start date: {start_date}, end_date: {end_date}')
    # Fetch the selected data range from the database
    conn = sqlite3.connect('../../data/model_results.db', timeout=15)
    c = conn.cursor()
    c.execute('''SELECT datetime, file_name, prediction, confidence, true_label 
                 FROM results 
                 WHERE datetime>=? 
                 AND datetime<=? 
                 AND (NOT true_label='none'
                 OR (true_label IS NULL AND NOT prediction='none'));''', 
                 (start_date, end_date))
    rows = c.fetchall()
    conn.close()
    result_dict = {l:[] for l in labels}
    for event in rows:
        dt, fn, pred, conf, true_label = event
        if true_label is not None:
            item = {'date':dt, 'image':fn, 'confidence':1.0, 'reviewed':True}
            result_dict[true_label].append(item)
        else:
            item = {'date':dt, 'image':fn, 'confidence': conf, 'reviewed':False}
            result_dict[pred].append(item)
    results = []
    for k, v in result_dict.items():
        results.append({'name': k, 'data':v})
    return jsonify(results)


@app.route('/api/serve_image/<string:img_fn>')
def serve_image(img_fn):
    img = cv.imread(f'../../imgs/{img_fn}')
    img = brighten_image(img)
    #img = histogram_equalize(img)
    img = Image.fromarray(img.astype('uint8'))
    # create file-object in memory
    file_object = io.BytesIO()
    # write PNG in file-object
    img.save(file_object, 'PNG')
    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG')