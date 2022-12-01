#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import sys

cap = cv.VideoCapture(sys.argv[1])
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Vertical scanlines
    for y in range(frame.shape[0] // 4):
        y = y * 4
        frame[y] = frame[y] * 0.5
    
    # Horizontal scanlines
    for x in range(frame.shape[1] // 4):
        x = x * 4
        frame[:, x] = frame[:, x] * 0.5

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()