from io import BytesIO
import cv2 as cv
import requests
import base64
from PIL import Image
from secrets import ENDPOINT

frame = cv.imread("../imgs/training/none_2020-12-06T09:50:05.jpg")
rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
pil_img = Image.fromarray(rgb_frame)
buf = BytesIO()
pil_img.save(buf, format="JPEG")
b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

r = requests.post(ENDPOINT, json={"b64_img":b64_img})
print(r)
results = r.json()
labels = results['labels']
confidence = [l[1] for l in results['confidence']]

print(labels)
print(confidence)
print(max(confidence))
print(1-max(confidence))