import base64
import cv2 as cv
import datetime as dt
import io
import json
import numpy as np
from PIL import Image
import sqlite3

from flask import Flask, redirect, render_template, request, jsonify, send_file
from pyinaturalist.rest_api import get_access_token, create_observation, update_observation, add_photo_to_observation

import os

INAT_USERNAME = os.getenv('INAT_USERNAME', '')
INAT_PASSWORD = os.getenv('INAT_PASSWORD', '')
INAT_APP_ID = os.getenv('INAT_APP_ID', '')
INAT_APP_SECRET = os.getenv('INAT_APP_SECRET', '')
DB_PATH = os.getenv('DB_PATH', '../data/model_results.db')


# Datetime format for printing and string representation
dt_fmt = '%Y-%m-%dT%H:%M:%S'

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

# Route for inaturalist api upload
@app.route('/api/inaturalist', methods=['POST'])
def inaturalist_api():
    json_data = json.loads(request.data)
    print(json_data)
    utc_key = json_data['utc_key']
    print(f'utc_key: {utc_key}')
    if utc_key is None:
        return
    print(f'key: {utc_key}')
    conn = sqlite3.connect(DB_PATH, timeout=15)
    query = '''SELECT datetime, file_name, prediction, true_label, inaturalist_id
               FROM results 
               WHERE utc_datetime = ?
               LIMIT 1;'''
    c = conn.cursor()
    c.execute(query, (utc_key,))
    row = c.fetchone()
    if row is None:
        return
    else:
        obs_timestamp = row[0]
        img_fn = row[1]
        pred_label = row[2]
        true_label = row[3]
        existing_inat_id = row[4]
    if true_label is not None:
        obs_label = true_label
    else:
        obs_label = pred_label

    # Get a token for the inaturalist API
    token = get_access_token(
        username=INAT_USERNAME,
        password=INAT_PASSWORD,
        app_id=INAT_APP_ID,
        app_secret=INAT_APP_SECRET,
    )

    latitude = 51.03128580819969
    longitude = -114.10264233236377
    positional_accuracy = 3
    obs_file_name = f'../../imgs/{img_fn}'

    species_map = {
        # Passer domesticus
        'sparrow': {
            'taxa_id': 13858,
        },
        # Poecile atricapillus
        'chickadee': {
            'taxa_id': 144815,
        },
        # Pica hudsonia
        'magpie': {
            'taxa_id': 143853,
        },
        # Sciurus carolinensis
        'squirrel': {
            'taxa_id': 46017,
        }
    }

    # Upload the observation to iNaturalist
    if existing_inat_id is None:
        # Check if there's an existing inat id within 5 minutes of this image
        # upload this image to that observation if so.
        window_timestamp = dt.datetime.fromisoformat(utc_key) - dt.timedelta(minutes=10)
        window_timestamp = window_timestamp.strftime(dt_fmt)
        query = '''SELECT inaturalist_id
                   FROM results 
                   WHERE utc_datetime <= :utc_dt
                   AND utc_datetime >= :prev_dt
                   AND inaturalist_id IS NOT NULL
                   AND (true_label = :lab OR (true_label IS NULL AND prediction = :lab))
                   ORDER BY utc_datetime DESC
                   LIMIT 1;'''
        c.execute(query, {'utc_dt':utc_key, 'prev_dt': window_timestamp, 'lab': obs_label})
        row = c.fetchone()
        if row is None or row[0] is None:
            response = create_observation(
                taxon_id=species_map[obs_label]['taxa_id'],
                observed_on_string=obs_timestamp,
                time_zone='Mountain Time (US & Canada)',
                description='Birb Cam image upload: https://github.com/evjrob/birbcam',
                tag_list=f'{obs_label}, Canada',
                latitude=latitude,
                longitude=longitude,
                positional_accuracy=positional_accuracy, # meters,
                access_token=token,
            )
            inat_observation_id = response[0]['id']
            print(f'No iNaturalist id found in previous ten minutes, creating new row with id {inat_observation_id}.')
        else:
            inat_observation_id = row[0]
            print(f'Found iNaturalist id in previous ten minutes, adding to id {inat_observation_id}.')
        # Upload the image captured
        r = add_photo_to_observation(
            inat_observation_id,
            access_token=token,
            photo=obs_file_name,
        )
        # Update the row in the database with the inaturalist id
        c.execute("UPDATE results SET inaturalist_id=? WHERE utc_datetime=?;", (inat_observation_id, utc_key))
    else:
        # This image had already been uploaded, we do not want to upload it again
        inat_observation_id = existing_inat_id
        print(f'Found existing iNaturalist id {inat_observation_id} for row, skipping.')
    
    conn.commit()
    conn.close()
    return jsonify({'inat_id': inat_observation_id})

# Route for model evaluation page
@app.route('/eval', methods=['GET'])
def model_evaluation_page():
    confidence = request.args.get('confidence', default=None)
    if confidence is None:
        confidence = 0.95
    else:
        confidence = float(confidence)
    prediction = request.args.get('prediction', default=None)
    if prediction is None:
        prediction = '%'
    print(prediction)
    # Fetch a single un-reviewed image from the database
    conn = sqlite3.connect(DB_PATH, timeout=15)
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

# Route for data revision page
@app.route('/revise', methods=['GET'])
def label_revise_page():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    c = conn.cursor()
    dt_key = request.args.get('dt_key', default=None)
    if dt_key is None:
        query = '''SELECT datetime
               FROM results 
               WHERE true_label IS NOT NULL
               ORDER BY utc_datetime DESC
               LIMIT 1;'''
        c.execute(query)
        dt_key = c.fetchone()[0]
    
    # Fetch a single un-reviewed image from the database
    query = '''SELECT utc_datetime, file_name, true_label
               FROM results 
               WHERE datetime = ?
               ORDER BY utc_datetime
               LIMIT 1;'''
    c.execute(query, (dt_key,))
    row = c.fetchone()
    conn.close()

    if row is not None:
        utc_key = row[0]
        img_fn = row[1]
        label = row[2]

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
        img_fn = None
        img_b64 = None
        label = None
        utc_key = None
    return render_template('revise.html', filename=img_fn, img=img_b64, 
                           label=label, utc_key=utc_key)

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
    conn = sqlite3.connect(DB_PATH, timeout=15)
    c = conn.cursor()
    c.execute("UPDATE results SET true_label=? WHERE utc_datetime=?;", (label_str, utc_key))
    conn.commit()
    conn.close()
    return redirect(request.referrer)

# Route for data loading API
@app.route('/api/data', methods=['POST'])
def get_data():
    labels = ['chickadee', 'magpie', 'sparrow', 'squirrel']
    json_data = request.get_json(force=True)
    start_date = json_data['start_date']
    end_date = json_data['end_date']
    print(f'start date: {start_date}, end_date: {end_date}')
    # Fetch the selected data range from the database
    conn = sqlite3.connect(DB_PATH, timeout=15)
    c = conn.cursor()
    c.execute('''SELECT utc_datetime, datetime, file_name, prediction, confidence, true_label, inaturalist_id
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
        utc_dt, dt, fn, pred, conf, true_label, inat_id = event
        if true_label is not None:
            true_labels = true_label.split(',')
            for tl in true_labels:
                item = {'utc_datetime':utc_dt, 
                    'date':dt, 
                    'image':fn, 
                    'confidence':1.0, 
                    'reviewed':True, 
                    'true_label':true_label,
                    'inat_id': inat_id}
                result_dict[tl].append(item)
        else:
            pred_labels = pred.split(',')
            for pl in pred_labels:
                item = {'utc_datetime':utc_dt, 
                    'date':dt, 
                    'image':fn, 
                    'confidence': conf, 
                    'reviewed':False, 
                    'true_label':pred_labels,
                    'inat_id': inat_id}
                result_dict[pl].append(item)
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

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)
