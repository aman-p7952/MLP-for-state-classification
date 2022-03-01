#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:46:59 2022

@author: aman
"""
import torch
import pandas as pd
from net import Net1
# key = {"standing": 0,
#        "squats": 1,
#        "None":2}
key = {"pushup up": 0,
        "pushup Down": 1,
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
df = pd.read_csv("highknees/highknees0.csv")

net=Net1()
net=net.double()
net.eval()
net.load_state_dict(torch.load("net.pt"))
predicted_state=[]


for i in range(len(df)):
    feature = torch.tensor([df.iloc[i, 1:133]])
    output=net(feature)
    _max, predicted = torch.max(output.data, 1)
    #print(output.data)
    if _max>0.5:
        out=get_key(predicted)
        predicted_state.append(out)
    else :
        out="None"
        predicted_state.append(out)
for i in range(132):
    df.drop(str(i),axis=1,inplace=True)
df["P_state"]=predicted_state
df.to_csv("data_pridiction9.csv",index=False)











   