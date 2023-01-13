#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import sys
import time
import math

cap = cv.VideoCapture(sys.argv[1])
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    chrom_strength = 32
    chrom_strength = abs(math.sin(time.time()*2)*chrom_strength)
    print(chrom_strength)
    chrom_strength = int(chrom_strength)

    out_frame = np.zeros((720, 1280, 3), np.uint8)
    # blue
    out_frame[:,chrom_strength:,0] = gray[:,:1280-chrom_strength]
    # green
    out_frame[:,:,1] = gray
    # red
    out_frame[:,:1280-chrom_strength,2] = gray[:,chrom_strength:]

    cv.imshow('frame', out_frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()