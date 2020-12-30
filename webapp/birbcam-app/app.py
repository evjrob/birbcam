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

def brighten_image(img, minimum_brightness=0.4):
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
@app.route('/eval', methods=['GET'])
def model_evaluation_page():
    confidence = request.args.get('confidence', default=None)
    if confidence is None:
        confidence = 1.0
    else:
        confidence = float(confidence)
    prediction = request.args.get('prediction', default=None)
    if prediction is None:
        prediction = '%'
    print(prediction)
    # Fetch a single un-reviewed image from the database
    conn = sqlite3.connect('../../data/model_results.db', timeout=15)
    query = '''SELECT utc_datetime, file_name, prediction, confidence
               FROM results 
               WHERE true_label IS NULL
               AND prediction LIKE ?
               AND confidence <= ?
               ORDER BY utc_datetime
               LIMIT 1;'''
    c = conn.cursor()
    c.execute(query, (prediction, confidence))
    row = c.fetchone()
    query = '''SELECT AVG(CASE WHEN true_label IS NULL THEN 0 ELSE 1 END) 
               FROM results
               WHERE prediction LIKE ?
               AND confidence <= ?;'''
    c.execute(query, (prediction, confidence))
    progress = c.fetchone()[0]
    if progress is None:
        progress = 100
    else:
        progress = progress * 100
    precision = '.2f'
    progress = f'{progress:{precision}}%'
    conn.close()

    if row is not None:
        show_eval = True
        utc_key = row[0]
        img_fn = row[1]
        label = row[2]
        label_conf = row[3]
        label_conf = f'{label_conf:{precision}}'

        # Read the image and histogram nomalize it
        img = cv.imread(f'../../imgs/{img_fn}')
        if img is None:
            print(utc_key)
            print(img_fn)
        
        adj = brighten_image(img)
        adj = cv.cvtColor(adj, cv.COLOR_RGB2BGR)

        # img io
        is_success, im_buf_arr = cv.imencode(".jpg", adj)
        png_img_bytes = im_buf_arr.tobytes()
        img_b64 = 'data:image/png;base64,' + base64.b64encode(png_img_bytes).decode('utf8')
    else:
        show_eval = False
        img_fn = None
        img_b64 = None
        label = None
        label_conf = None
        utc_key = None
    return render_template('evaluate.html', prediction=prediction, confidence=confidence, 
                           progress=progress, show_eval=show_eval, filename=img_fn, 
                           img=img_b64, label=label, label_conf=label_conf, utc_key=utc_key)

# Route for model evaluation page
@app.route('/api/model_eval_submit', methods=['POST'])
def model_evaluation_api():
    utc_key = str(request.form['utc_key'])
    labels = request.form.getlist('label')
    if len(labels) == 0:
        labels = ['none']
    if 'other' in labels:
        other_label = str(request.form['othertext'])
        labels[labels.index('other')] = other_label
    label_str = ','.join(labels)
    print(f'key: {utc_key}, label: {label_str}')
    # Update the row in the database with the selected true label
    conn = sqlite3.connect('../../data/model_results.db', timeout=15)
    c = conn.cursor()
    c.execute("UPDATE results SET true_label=? WHERE utc_datetime=?;", (label_str, utc_key))
    conn.commit()
    conn.close()
    return redirect(request.referrer)

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
            true_labels = true_label.split(',')
            for tl in true_labels:
                item = {'date':dt, 'image':fn, 'confidence':1.0, 'reviewed':True}
                result_dict[tl].append(item)
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
