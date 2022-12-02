#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import sys
from wand.image import Image

def libraries(frame):
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

    return frame

def plain(orig_frame):
    # Downsample to make processing faster
    orig_frame = cv.pyrDown(orig_frame)

    frame = np.zeros(orig_frame.shape, np.uint8) 

    for y in range(orig_frame.shape[0]):
        for x in range(orig_frame.shape[1]):
            (u, v) = (x/orig_frame.shape[1], y/orig_frame.shape[0])
            (dist_x, dist_y) = (0.5 - u, 0.5 - v)

            strength = -.5
            new_x = u - dist_y * dist_y * dist_x * strength/(orig_frame.shape[1]/orig_frame.shape[0]) 
            new_y = v - dist_x * dist_x * dist_y * strength

            new_x *= orig_frame.shape[1]
            new_y *= orig_frame.shape[0]

            #print(f"({x}, {y}) -> ({new_x}, {new_y})")

            new_x = int(new_x)
            new_y = int(new_y)

            if new_x >= 0 and new_x < orig_frame.shape[1] and new_y >= 0 and new_y < orig_frame.shape[0]:
                frame[new_y, new_x] = orig_frame[y, x]

    return frame

cap = cv.VideoCapture(sys.argv[1])
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #frame = libraries(frame)
    frame = plain(frame)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()