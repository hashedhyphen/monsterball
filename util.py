#!/usr/bin/env python
import cv2

ball = cv2.imread("monsterball.jpg")

# detect the contour of the monster ball
ball_gray   = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)
_, thresh   = cv2.threshold(ball_gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# create a mask by filling the area (contours[1]) in black
cv2.drawContours(ball_gray, contours, 1, 0, -1)
_, mask = cv2.threshold(ball_gray, 127, 255, cv2.THRESH_BINARY) # for sharpness

def draw_balls(frame, rects):
    for x1, y1, x2, y2 in rects:
        roi = frame[y1:y2, x1:x2]
        resized_ball = cv2.resize(ball, (x2 - x1, y2 - y1))
        resized_mask = cv2.resize(mask, (x2 - x1, y2 - y1))
        resized_mask_inv = cv2.bitwise_not(resized_mask)

        # black-out the place where the monster ball should appear
        frame_bg = cv2.bitwise_and(roi, roi, mask=resized_mask)

        # extract the monster ball itself
        resized_ball_fg = cv2.bitwise_and(resized_ball, resized_ball,
                                          mask=resized_mask_inv)

        # embed the monster ball into the frame
        dst = cv2.add(frame_bg, resized_ball_fg)
        frame[y1:y2, x1:x2] = dst
