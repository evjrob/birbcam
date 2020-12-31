import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import medfilt2d

# Open webcam
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FPS, 1)

# Set up Background Subtraction Model (bgsub)
backSub = cv.createBackgroundSubtractorMOG2()

# Bgsub and median filtering parameters
mask_thresh = 255
kernel_size = 25
lr = 0.05
burn_in = 30
i = 0

# Loop through the frames from the webcam
while True:
    ret, frame = capture.read()
    fgMask = backSub.apply(frame, learningRate=lr)
    # Avoid early false positives
    if i < burn_in:
        i += 1
        continue
    # Threshold mask - plot when change detected
    fgMaskMedian = medfilt2d(fgMask, kernel_size)
    if (fgMaskMedian >= mask_thresh).any():
        fig, axs = plt.subplots(1, 3)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        axs[0].imshow(frame)
        axs[1].imshow(fgMask)
        axs[2].imshow(fgMaskMedian)
        plt.show()
