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
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,100,200)
    cv.imshow('frame', edges)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()