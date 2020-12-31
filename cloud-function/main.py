import base64
from fastai.vision.all import *
from google.cloud import storage
import numpy as np

# Download model artifact to /tmp
storage_client = storage.Client(project='birbcam')
bucket = storage_client.get_bucket('birbcam')
blob = bucket.blob('models/birbcam_prod.pkl')
blob.download_to_filename('/tmp/birbcam_prod.pkl')
learn = load_learner('/tmp/birbcam_prod.pkl')

def model_inference(request):
    """Responds to birbcam model HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        model prediction dictionary for the passed image
    """
    request_json = request.get_json()
    if request.args and 'b64_img' in request.args:
        img_data = base64.b64decode(request.args.get('b64_img'))
    elif request_json and 'b64_img' in request_json:
        img_data = base64.b64decode(request_json['b64_img'])
    else:
        return "Bad request!"

    img = np.asarray(Image.open(io.BytesIO(img_data)))
    pred = learn.predict(img)
    labels = pred[0]
    confidence = pred[2]
    result = {
        "labels": list(labels),
        "confidence": list(zip(list(learn.dls.vocab), confidence.tolist()))
    }
    
    return result