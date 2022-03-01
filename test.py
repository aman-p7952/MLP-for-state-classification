#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 20:16:57 2022
@author: aman
"""
import torch
import cv2
import mediapipe as mp
import numpy as np
import torch.nn as nn
from net import Net1
import pandas as pd
df = pd.read_csv("data.csv")
num_classes = 7
state="null"
key = {"pushup up": 0,
        "pushup down": 1,
        "None":2,
        "standing": 3,
        "squats": 4,
        "left knee high":5,
        "right knee high":6,}
def get_key(val):
    for k, value in key.items():
         if val == value:
             return k
    return "key doesn't exist"
predicted=-1
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(2)

net=Net1()
net.eval()
net.load_state_dict(torch.load("net.pt"))
i=0
with mp_pose.Pose(
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
      # break
    height,width,_=image.shape
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    row=[]
    i+=1
    try:
        for j in range(33):
            x = results.pose_landmarks.landmark[j].x
            y = results.pose_landmarks.landmark[j].y
            z = results.pose_landmarks.landmark[j].z
            v = results.pose_landmarks.landmark[j].visibility
            row.append(x)
            row.append(y)
            row.append(z)
            row.append(v)
        output=net(torch.tensor([row]))
        predicted = torch.argmax(output.data, 1)
        state=get_key(predicted)
    except Exception as e:
        print(e)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    image=cv2.flip(image, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,state,
                (100, 100), font,
                3, (0, 0, 255),
                3, cv2.LINE_AA)
    
    cv2.imshow('MediaPipe Pose',image )
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()