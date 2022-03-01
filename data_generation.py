#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 14:02:20 2022

@author: aman
"""
import os
from media_util import *
import cv2
import csv
import time
import os
import argparse
import mediapipe as mp
import numpy as np
from media_util import angle, isin, isin_frontview, isin_leftview, normal, facing, is_squats, is_pushup, torso_length, angle_xy, is_facingup

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('../combined_label.avi', fourcc, 20.0, (640,480))

# parser = argparse.ArgumentParser()
# parser.add_argument("file")
# args = parser.parse_args()
# file=args.file
start_time = time.time()
file = "/home/aman/sporty/project1/combined.avi"
cap = cv2.VideoCapture("pushup/push1.mp4")
prv_state = -1
state = "ZZ"
meassage = None
count = 0
ratio = -1
angle1 = -1
nor = [0, 0, 0]
torso = 0
per = 0
angle_le = 0
start = False
start1 = False
name = None
list_state = []
i = 0


header=[str(i) for i in range(132)]
header=["frame"]+header+["state"]
if os.path.isfile(".pushup//data0.csv"):
    f = open('pushup/data0.csv', 'a', encoding='UTF8')
    writer = csv.writer(f)

else:
    f = open('pushup/data0.csv', 'w', encoding='UTF8')
    writer = csv.writer(f)
    writer.writerow(header)
with mp_pose.Pose(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            # continue
            break
        height, width, _ = image.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        i += 1
        try:
            per = isin(results, threshold=0.4)
            torso = torso_length(results, height, width)
            angle_lk = angle(results, 23, 25, 27, height=height, width=width)
            angle_rk = angle(results, 24, 26, 28, height=height, width=width)
            angle_lh = angle(results, 11, 23, 25, height=height, width=width)
            angle_rh = angle(results, 12, 24, 26, height=height, width=width)
            angle_lk_xy = angle_xy(results, 23, 25, 27,
                                   height=height, width=width)
            angle_rk_xy = angle_xy(results, 24, 26, 28,
                                   height=height, width=width)
            angle_lh_xy = angle_xy(results, 11, 23, 25,
                                   height=height, width=width)
            angle_rh_xy = angle_xy(results, 12, 24, 26,
                                   height=height, width=width)
            angle_lk = angle_xy(results, 23, 25, 27,
                                height=height, width=width)
            angle_rk = angle(results, 24, 26, 28, height=height, width=width)
            angle_lh = angle_xy(results, 11, 23, 25,
                                height=height, width=width)
            angle_rh = angle(results, 12, 24, 26, height=height, width=width)
            angle_le = angle_xy(results, 11, 13, 15, height, width)
            angle_le_xy = angle_xy(results, 11, 13, 15, height, width)
            angle_re = angle(results, 12, 14, 16, height, width)
            angle_re_xy = angle_xy(results, 12, 14, 16, height, width)
            angle_ls = angle_xy(results, 13, 11, 23, height, width)
            angle_rs = angle_xy(results, 14, 12, 24, height, width)
            row = []
            list_state = []
            row.append(str(i))
            for j in range(33):
                x = results.pose_landmarks.landmark[j].x
                y = results.pose_landmarks.landmark[j].y
                z = results.pose_landmarks.landmark[j].z
                v = results.pose_landmarks.landmark[j].visibility
                row.append(str(x))
                row.append(str(y))
                row.append(str(z))
                row.append(str(v))
            # squats
            if not is_pushup(results, height, width, th=4) and isin(results, 0.4) == 100 and angle_lk > 110 and\
                    angle_rk > 110 and angle_lh > 100 and angle_rh > 100 and knee_position(results) and is_standing(results):
                state = "U"
                name = "standing"
                list_state.append(name)
            elif not is_pushup(results, height, width, th=4) and isin(results, 0.4) == 100 and angle_rh < 130 and angle_lh < 130 \
                    and angle_rk < 130 and angle_lh < 130 and knee_position(results) and is_standing(results):
                state = "V"
                name = "squats"
                list_state.append(name)

            else:
                state = "ZZ"
                name = "None"
                list_state.append(name)
            row.append(name)
            writer.writerow(row)
        except Exception as e:
            # print(".........")
            print(e)
            message = None
        # x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
        # y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        # arr.append((x,y))
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,
                    # str(round(angle1)),
                    str(list_state),
                    (100, 100), font,
                    0.7, (0, 0, 255),
                    1, cv2.LINE_AA)
        cv2.imshow('MediaPipe Pose', image)
        if not os.path.isdir('./images/all'):
            os.mkdir('./images/all')
        filename = "./images/all/all"+str(i)+".jpg"
        cv2.imwrite(filename, image)
        # if len(list_state)!=0:
        #     writer.writerow([str(i),str(list_state),str(len(list_state))])
        # if state=="V":
        #     if not os.path.isdir( './images/squats' ) :
        #         os.mkdir( './images/squats' )

        #     filename="./images/squats/squats"+str(i)+".jpg"
        #     cv2.imwrite(filename, image)
        # if state=="D":
        #     if not os.path.isdir( './images/high_knees' ) :
        #         os.mkdir( './images/high_knees' )
        #     filename="./images/high_knees/left"+str(i)+".jpeg"
        #     cv2.imwrite(filename, image)
        # if state=="E":
        #     if not os.path.isdir( './images/high_knees' ) :
        #         os.mkdir( './images/high_knees' )
        #     filename="./images/high_knees/right"+str(i)+".jpeg"
        #     cv2.imwrite(filename, image)
        # out.write(image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
# out.release()
f.close()
