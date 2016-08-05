#!/usr/bin/env python
import numpy as np
import cv2
import sys

from core import detect
from util import draw_balls

cam = cv2.VideoCapture(1)
if not cam.isOpened():
    sys.exit("Couldn't open the default camera!")

while True:
    _, frame = cam.read()
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detect(gray)

    vis = frame.copy()
    draw_balls(vis, rects)

    cv2.imshow("output", vis)

    if 0xFF & cv2.waitKey(5) == ord("q"):
        break

cv2.destroyAllWindows()
