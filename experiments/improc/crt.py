#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import sys
from wand.image import Image

cap = cv.VideoCapture(sys.argv[1])
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Vertical scanlines
    INTERVAL = 32
    HEIGHT = 8
    for y in range(frame.shape[0] // INTERVAL):
        y = y * INTERVAL
        for i in range(HEIGHT):
            frame[y + i] = frame[y + i] * 0.9

    # Downsample
    frame = cv.pyrDown(cv.pyrDown(frame))
    # Upsample to keep the same size output feed
    # (while lowering quality)
    frame = cv.pyrUp(cv.pyrUp(frame))

    # Distort
    wand_frame = Image.from_array(frame)
    wand_frame.virtual_pixel = 'transparent'
    wand_frame.distort('barrel', (0,0,0.1,0.9))
    frame = np.array(wand_frame)

    """
    
    # Horizontal scanlines
    for x in range(frame.shape[1] // 4):
        x = x * 4
        frame[:, x] = frame[:, x] * 0.5
    """

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()