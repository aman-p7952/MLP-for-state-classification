#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 07:44:20 2022

@author: aman
"""

import numpy as np


def normal(results, p1, p2, p3, height, width):
    a_x = results.pose_landmarks.landmark[p1].x*width
    a_y = results.pose_landmarks.landmark[p1].y*height
    a_z = results.pose_landmarks.landmark[p1].z*width
    b_x = results.pose_landmarks.landmark[p2].x*width
    b_y = results.pose_landmarks.landmark[p2].y*height
    b_z = results.pose_landmarks.landmark[p2].z*width
    c_x = results.pose_landmarks.landmark[p3].x*width
    c_y = results.pose_landmarks.landmark[p3].y*height
    c_z = results.pose_landmarks.landmark[p3].z*width
    a = np.array([a_x, a_y, a_z])
    b = np.array([b_x, b_y, b_z])
    c = np.array([c_x, c_y, c_z])
    ba = a - b
    bc = c - b
    cross = np.cross(bc, ba)
    vec = cross/np.linalg.norm(cross)
    x = (np.arccos(vec[0]/np.linalg.norm(vec)))*180/np.pi
    # y = np.degrees(np.arccos(vec[1]/np.linalg.norm(vec)))
    # z = np.degrees(np.arccos(vec[2]/np.linalg.norm(vec)))
    y = (np.arccos(vec[1]/np.linalg.norm(vec)))*180/np.pi
    z = (np.arccos(vec[2]/np.linalg.norm(vec)))*180/np.pi
    return [round(x), round(y), round(z)]


def angle(results, p1, p2, p3, height, width):
    a_x = results.pose_landmarks.landmark[p1].x*width
    a_y = results.pose_landmarks.landmark[p1].y*height
    a_z = results.pose_landmarks.landmark[p1].z*width
    b_x = results.pose_landmarks.landmark[p2].x*width
    b_y = results.pose_landmarks.landmark[p2].y*height
    b_z = results.pose_landmarks.landmark[p2].z*width
    c_x = results.pose_landmarks.landmark[p3].x*width
    c_y = results.pose_landmarks.landmark[p3].y*height
    c_z = results.pose_landmarks.landmark[p3].z*width
    a = np.array([a_x, a_y, a_z])
    b = np.array([b_x, b_y, b_z])
    c = np.array([c_x, c_y, c_z])
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle1 = np.arccos(cosine_angle)
    return np.degrees(angle1)


def angle_none(results, p1, p2, p3, height, width, threshold=0.5):
    a_x = results.pose_landmarks.landmark[p1].x*width
    a_y = results.pose_landmarks.landmark[p1].y*height
    a_z = results.pose_landmarks.landmark[p1].z*width
    b_x = results.pose_landmarks.landmark[p2].x*width
    b_y = results.pose_landmarks.landmark[p2].y*height
    b_z = results.pose_landmarks.landmark[p2].z*width
    c_x = results.pose_landmarks.landmark[p3].x*width
    c_y = results.pose_landmarks.landmark[p3].y*height
    c_z = results.pose_landmarks.landmark[p3].z*width
    a = np.array([a_x, a_y, a_z])
    b = np.array([b_x, b_y, b_z])
    c = np.array([c_x, c_y, c_z])
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle1 = np.arccos(cosine_angle)
    if results.pose_landmarks.landmark[p1].visibility > threshold and \
            results.pose_landmarks.landmark[p2].visibility > threshold and\
            results.pose_landmarks.landmark[p3].visibility > threshold:
        return np.degrees(angle1)
    else:
        return None
    # return np.degrees(angle1)


def angle_xy(results, p1, p2, p3, height, width, threshold=0.5):
    a_x = results.pose_landmarks.landmark[p1].x*width
    a_y = results.pose_landmarks.landmark[p1].y*height
    a_z = results.pose_landmarks.landmark[p1].z*width
    b_x = results.pose_landmarks.landmark[p2].x*width
    b_y = results.pose_landmarks.landmark[p2].y*height
    b_z = results.pose_landmarks.landmark[p2].z*width
    c_x = results.pose_landmarks.landmark[p3].x*width
    c_y = results.pose_landmarks.landmark[p3].y*height
    c_z = results.pose_landmarks.landmark[p3].z*width
    a = np.array([a_x, a_y])
    b = np.array([b_x, b_y])
    c = np.array([c_x, c_y])
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle1 = np.arccos(cosine_angle)
    # if results.pose_landmarks.landmark[p1].visibility>threshold and \
    #     results.pose_landmarks.landmark[p2].visibility>threshold and\
    #          results.pose_landmarks.landmark[15].visibility>threshold:
    #              return np.degrees(angle1)
    # else:
    #     return None
    return np.degrees(angle1)


def isin(results, threshold):
    inside = [1 for i in range(
        33) if results.pose_landmarks.landmark[i].visibility > threshold]
    return round(len(inside)*100/33)
def isin_lr(results,threshold):
    count =0
    if results.pose_landmarks.landmark[0].visibility>threshold :
        count+=1
    if results.pose_landmarks.landmark[11].visibility>threshold or results.pose_landmarks.landmark[12].visibility>threshold:
        count+=1
    if results.pose_landmarks.landmark[13].visibility>threshold or results.pose_landmarks.landmark[14].visibility>threshold:
        count+=1
    if results.pose_landmarks.landmark[15].visibility>threshold or results.pose_landmarks.landmark[16].visibility>threshold:
        count+=1
    if results.pose_landmarks.landmark[25].visibility>threshold or results.pose_landmarks.landmark[26].visibility>threshold:
        count+=1
    if results.pose_landmarks.landmark[23].visibility>threshold or results.pose_landmarks.landmark[24].visibility>threshold:
        count+=1
    if results.pose_landmarks.landmark[27].visibility>threshold or results.pose_landmarks.landmark[28].visibility>threshold:
        count+=1
    return count*100/7

def isin_frontview(results, threshold):
    count = 0
    # if results.pose_landmarks.landmark[1].visibility>threshold:
    #     count+=1
    inside = [1 for i in range(
        11, 17) if results.pose_landmarks.landmark[i].visibility > threshold]
    count = count+sum(inside)
    return count*100/6


def isin_leftview(results, threshold):
    count = 0
    if results.pose_landmarks.landmark[15].visibility > threshold:
        count += 1
    if results.pose_landmarks.landmark[13].visibility > threshold:
        count += 1
    if results.pose_landmarks.landmark[11].visibility > threshold:
        count += 1
    if results.pose_landmarks.landmark[23].visibility > threshold:
        count += 1
    if results.pose_landmarks.landmark[25].visibility > threshold:
        count += 1
    if results.pose_landmarks.landmark[27].visibility > threshold:
        count += 1
    return count*100/6


def isin_rightview(results, threshold):
    count = 0
    if results.pose_landmarks.landmark[16].visibility > threshold:
        count += 1
    if results.pose_landmarks.landmark[14].visibility > threshold:
        count += 1
    if results.pose_landmarks.landmark[12].visibility > threshold:
        count += 1
    if results.pose_landmarks.landmark[24].visibility > threshold:
        count += 1
    if results.pose_landmarks.landmark[26].visibility > threshold:
        count += 1
    if results.pose_landmarks.landmark[28].visibility > threshold:
        count += 1
    return count*100/6


def isin_squatsview(results, threshold):

    inside = [1 for i in range(
        23, 29) if results.pose_landmarks.landmark[i].visibility > threshold]
    count = sum(inside)
    if results.pose_landmarks.landmark[11].visibility > threshold:
        count += 1
    if results.pose_landmarks.landmark[12].visibility > threshold:
        count += 1
    return count*100/8


def facing(results, height, width):
    right_shoulder_x = results.pose_landmarks.landmark[12].x*width
    left_shoulder_x = results.pose_landmarks.landmark[11].x*width
    visibility_rs = results.pose_landmarks.landmark[12].visibility
    visibility_ls = results.pose_landmarks.landmark[11].visibility
    # right_shoulder_z=results.pose_landmarks.landmark[12].z*width
    # left_shoulder_z=results.pose_landmarks.landmark[11].z*width
    nor = normal(results, 12, 11, 23, height=height, width=width)
    if visibility_rs > 0.9 and visibility_ls > 0.9:
        if right_shoulder_x < left_shoulder_x:
            if nor[0] > 30 and nor[0] < 150:
                return "front"
            elif nor[0] > 150:
                return "Right"
            else:
                return "left"
        else:
            return "back"
    else:
        return "0"


def is_squats(results, height, width):

    THD = 60
    angle_lh = angle(results, 23, 11, 13, height=height, width=width)
    angle_rh = angle(results, 24, 12, 14, height=height, width=width)
    angle_ll = angle(results, 23, 25, 27, height=height, width=width)
    left_foot_index_x = results.pose_landmarks.landmark[31].x*width
    left_wrist_x = results.pose_landmarks.landmark[15].x*width
    right_wrist_x = results.pose_landmarks.landmark[16].x*width
    left_wrist_y = results.pose_landmarks.landmark[15].y*height
    right_wrist_y = results.pose_landmarks.landmark[16].y*height
    left_wrist_z = results.pose_landmarks.landmark[15].z*width
    right_wrist_z = results.pose_landmarks.landmark[16].z*width
    left_shoulder_x = results.pose_landmarks.landmark[11].x*width
    right_shoulder_x = results.pose_landmarks.landmark[12].x*width
    left_shoulder_y = results.pose_landmarks.landmark[11].y*height
    right_shoulder_y = results.pose_landmarks.landmark[12].y*height
    left_shoulder_z = results.pose_landmarks.landmark[11].z*width
    right_shoulder_z = results.pose_landmarks.landmark[12].z*width
    wrist_distance = np.linalg.norm([left_wrist_x-right_wrist_x, left_wrist_y-right_wrist_y,
                                     left_wrist_z-right_wrist_z])
    shoulder_distance = np.linalg.norm([left_shoulder_x-right_shoulder_x, left_shoulder_y-right_shoulder_y,
                                        left_shoulder_z-right_shoulder_z])
    if angle_lh > THD and angle_rh > THD and (wrist_distance < 2*shoulder_distance) and\
            angle_ll < 100:
        return True
    else:
        return False


def is_pushup(results, height, width, th):
    left_shoulder_x = results.pose_landmarks.landmark[11].x*width
    left_shoulder_y = results.pose_landmarks.landmark[11].y*height
    left_shoulder_z = results.pose_landmarks.landmark[11].z*width
    left_hip_x = results.pose_landmarks.landmark[23].x*width
    left_hip_y = results.pose_landmarks.landmark[23].y*height
    left_hip_z = results.pose_landmarks.landmark[23].z*width
    distance_left = np.linalg.norm([left_shoulder_x-left_hip_x, left_shoulder_y-left_hip_y,
                                    left_shoulder_z-left_hip_z])
    right_shoulder_x = results.pose_landmarks.landmark[12].x*width
    right_shoulder_y = results.pose_landmarks.landmark[12].y*height
    right_shoulder_z = results.pose_landmarks.landmark[12].z*width
    right_hip_x = results.pose_landmarks.landmark[24].x*width
    right_hip_y = results.pose_landmarks.landmark[24].y*height
    right_hip_z = results.pose_landmarks.landmark[24].z*width
    distance_right = np.linalg.norm([right_shoulder_x-right_hip_x, right_shoulder_y-right_hip_y,
                                     right_shoulder_z-right_hip_z])
    distance = abs(right_shoulder_x-left_shoulder_x)
    right_wrist_y = results.pose_landmarks.landmark[16].y*height
    left_wrist_y = results.pose_landmarks.landmark[15].y*height
    # if isin_frontview(results, 0.9)==100:
    #     if (distance*0.97<np.abs(left_hip_z-left_shoulder_z)):
    #         return True
    #     else:
    #         return False
    if isin_leftview(results, threshold=0.7) >= 100:
        if abs(left_shoulder_y-left_hip_y) < distance_left/th:
            return True
        else:
            return False
    elif isin_rightview(results, threshold=0.8) == 100:
        if abs(right_shoulder_y-right_hip_y) < distance_right/th:
            return True
        else:
            return False
    # elif isin_frontview(results, 0.8) == 100:
    #     if abs(left_hip_y-left_shoulder_y) < distance/th and \
    #             abs(left_wrist_y-right_wrist_y) < 50:
    #         return True
    #     else:
    #         return False
    else:
        return False
    

def is_facingup(results, height, width):
    left_shoulder_x = results.pose_landmarks.landmark[11].x*width
    left_shoulder_y = results.pose_landmarks.landmark[11].y*height
    left_shoulder_z = results.pose_landmarks.landmark[11].z*width
    left_hip_x = results.pose_landmarks.landmark[23].x*width
    left_hip_y = results.pose_landmarks.landmark[23].y*height
    left_hip_z = results.pose_landmarks.landmark[23].x*width
    left_ear_y = results.pose_landmarks.landmark[7].y*height
    nose_y = results.pose_landmarks.landmark[0].y*height
    distance = np.linalg.norm([left_shoulder_x-left_hip_x, left_shoulder_y-left_hip_y,
                               left_shoulder_z-left_hip_z])
    # if isin_frontview(results, 0.9)==100:
    #     if (distance*0.97<np.abs(left_hip_z-left_shoulder_z)):
    #         return True
    #     else:
    #         return False
    
    # if isin_leftview(results, threshold=0.8) == 100:
    #     if abs(left_shoulder_y-left_hip_y) < distance/5 and left_ear_y > nose_y:
    #         return True
    #     else:
    #         return False

    if isin_leftview(results, threshold=0.8) == 100:
        if left_ear_y > nose_y:
            return True
        else:
            return False
    else :
        return False
def torso_length(results, height, width):
    left_shoulder_x = results.pose_landmarks.landmark[11].x*width
    left_shoulder_y = results.pose_landmarks.landmark[11].y*height
    left_shoulder_z = results.pose_landmarks.landmark[11].z*width

    right_shoulder_x = results.pose_landmarks.landmark[12].x*width
    right_shoulder_y = results.pose_landmarks.landmark[12].y*height
    right_shoulder_z = results.pose_landmarks.landmark[12].z*width

    left_hip_x = results.pose_landmarks.landmark[23].x*width
    left_hip_y = results.pose_landmarks.landmark[23].y*height
    left_hip_z = results.pose_landmarks.landmark[23].x*width

    right_hip_x = results.pose_landmarks.landmark[24].x*width
    right_hip_y = results.pose_landmarks.landmark[24].y*height
    right_hip_z = results.pose_landmarks.landmark[24].x*width

    left_shoulder = np.array([left_shoulder_x, left_shoulder_y])
    right_shoulder = np.array([right_shoulder_x, right_shoulder_y])

    left_hip = np.array([left_hip_x, left_hip_y])
    right_hip = np.array([right_hip_x, right_hip_y])
    mid_hip = (left_hip+right_hip)/2
    mid_shoulder = (left_shoulder+right_shoulder)/2
    length = np.linalg.norm(mid_hip-mid_shoulder)
    return length


def knee_position(results):
    left_knee_x = results.pose_landmarks.landmark[25].x
    left_knee_y = results.pose_landmarks.landmark[25].y
    left_knee_z = results.pose_landmarks.landmark[25].z
    right_knee_x = results.pose_landmarks.landmark[26].x
    right_knee_y = results.pose_landmarks.landmark[26].y
    right_knee_z = results.pose_landmarks.landmark[26].z
    left_ankle_x = results.pose_landmarks.landmark[27].x
    left_ankle_y = results.pose_landmarks.landmark[27].y
    left_ankle_z = results.pose_landmarks.landmark[27].z
    
    leg_height = np.linalg.norm([left_ankle_x-left_knee_x,left_ankle_y-left_knee_y,left_ankle_z-left_knee_z])
    
    if abs(left_knee_y-right_knee_y) < 0.2*leg_height:
        return True
    else:
        return False


def knee_ankle_distance(results):
    left_knee_y = results.pose_landmarks.landmark[25].y
    right_knee_y = results.pose_landmarks.landmark[26].y
    left_ankle_y = results.pose_landmarks.landmark[27].y
    right_ankle_y = results.pose_landmarks.landmark[28].y
    left = left_ankle_y-left_knee_y
    right = right_ankle_y-right_knee_y
    if abs(left-right)/(left+right) < 0.25:
        return True
    else:
        return False


def is_standing(results):
    left_shoulder_y = results.pose_landmarks.landmark[11].y
    right_shoulder_y = results.pose_landmarks.landmark[12].y
    left_hip_y = results.pose_landmarks.landmark[23].y
    right_hip_y = results.pose_landmarks.landmark[24].y
    left_knee_y = results.pose_landmarks.landmark[25].y
    right_knee_y = results.pose_landmarks.landmark[26].y
    left_ankle_y = results.pose_landmarks.landmark[27].y
    right_ankle_y = results.pose_landmarks.landmark[28].y
    # if left_ankle_y>left_knee_y and left_knee_y>left_hip_y and left_hip_y>left_shoulder_y and\
    #     right_ankle_y>right_knee_y and right_knee_y>right_hip_y and right_hip_y>right_shoulder_y:
    #     return
    if left_ankle_y > left_knee_y and left_hip_y > left_shoulder_y and left_knee_y > left_shoulder_y and\
            right_ankle_y > right_knee_y and right_hip_y > right_shoulder_y and right_hip_y > right_shoulder_y:
        return True
    else:
        return False


def knee_horizontal(results, th):
    left_knee_x = results.pose_landmarks.landmark[25].x
    left_knee_y = results.pose_landmarks.landmark[25].y
    left_knee_z = results.pose_landmarks.landmark[25].z
    left_ankle_x = results.pose_landmarks.landmark[27].x
    left_ankle_y = results.pose_landmarks.landmark[27].y
    left_ankle_z = results.pose_landmarks.landmark[27].z
    distance_left = np.linalg.norm([left_knee_x-left_ankle_x, left_knee_y-left_ankle_y,
                                    left_knee_z-left_ankle_z])
    right_knee_x = results.pose_landmarks.landmark[26].x
    right_knee_y = results.pose_landmarks.landmark[26].y
    right_knee_z = results.pose_landmarks.landmark[26].z
    right_ankle_x = results.pose_landmarks.landmark[28].x
    right_ankle_y = results.pose_landmarks.landmark[28].y
    right_ankle_z = results.pose_landmarks.landmark[28].z
    distance_right = np.linalg.norm([right_knee_x-right_ankle_x, right_knee_y-right_ankle_y,
                                     right_knee_z-right_ankle_z])
    if abs(left_knee_y-left_ankle_y) < distance_left/th:
        return True
    else:
        return False


def ankles_apart(results, th=2):
    left_knee_x = results.pose_landmarks.landmark[25].x
    left_knee_y = results.pose_landmarks.landmark[25].y
    left_ankle_x = results.pose_landmarks.landmark[27].x
    left_ankle_y = results.pose_landmarks.landmark[27].y
    right_ankle_x = results.pose_landmarks.landmark[28].x
    left_leg = np.linalg.norm(
        [left_ankle_x-left_knee_x, left_ankle_y-left_knee_y])
    if abs(right_ankle_x-left_ankle_x) > th*left_leg:
        return True
    else:
        return False
