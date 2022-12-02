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
            # Get X and Y coordinates in a range of 0.0 to 1.0
            (u, v) = (x/orig_frame.shape[1], y/orig_frame.shape[0])
            # Distance to the center
            # (positive if left/up from center, negative if right/down from center)
            (dist_x, dist_y) = (0.5 - u, 0.5 - v)

            # How much distortion to apply
            strength_y = .5
            # The x dimension is larger than y, corect for that by reducing the strength
            strength_x = strength_y / (orig_frame.shape[1] / orig_frame.shape[0])
            new_x = u + dist_y * dist_y * dist_x * strength_x            
            new_y = v + dist_x * dist_x * dist_y * strength_y

            # Scale back up to pixel coordinates
            new_x *= orig_frame.shape[1]
            new_y *= orig_frame.shape[0]

            # float to int so we can use them as indices
            new_x = int(new_x)
            new_y = int(new_y)

            # Because new_x and new_y are warped towards the center, 
            # they are always within bounds
            frame[new_y, new_x] = orig_frame[y, x]

    return frame

def barrel_map(x, y, frame_shape):
    size_y = frame_shape[0]
    size_x = frame_shape[1]
    # Distance to the center
    # (positive if left/up from center, negative if right/down from center)
    dist_x = size_x//2 - x
    dist_y = size_y//2 - y

    # Relative pixel movement
    off_x = (dist_y * dist_y) * dist_x
    off_y = (dist_x * dist_x) * dist_y
    # distances are not normalize, so correct the values by dividing by
    # (image width * image_height), cancelling out scaling of dist_x*dist_y,
    # a common factor of both computations above.
    size_xy = size_x * size_y 
    off_x = off_x // size_xy
    off_y = off_y // size_xy

    # Reduce the strength of the effect.
    # the frame is wider than it is high, yet we want the same strength.
    off_y = off_y // 2         # strength = 1/2
    off_x = off_x * 9 // 32    # strength = 1/2 / (1280/720) = 9/32

    new_x = x + off_x
    new_y = y + off_y

    return (new_x, new_y)

def plain_int(orig_frame):
    # Downsample to make processing faster
    orig_frame = cv.pyrDown(orig_frame)

    frame = np.zeros(orig_frame.shape, np.uint8) 

    for y in range(orig_frame.shape[0]):
        for x in range(orig_frame.shape[1]):
            (new_x, new_y) = barrel_map(x, y, orig_frame.shape)

            frame[new_y, new_x] = orig_frame[y, x]
    return frame

def run_filter():
    cap = cv.VideoCapture(sys.argv[1])
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = plain_int(frame)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def find_empty_cells():
    mapped_cells = set()
    for x in range(1280):
        for y in range(720):
            mapped_cells.add(barrel_map(x, y, (720, 1280)))
    print(1280*720)
    print(len(mapped_cells))

if __name__ == "__main__":
    find_empty_cells()