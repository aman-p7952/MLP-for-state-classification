#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:30:52 2022

@author: aman
"""
from collections import deque
import pandas as pd
df = pd.read_csv("./data_pridiction7.csv")
df=df["P_state"]
l=list(df)
window=5
previous_state="None"
count=0
dq=deque([],3)

def check_prev(i,window,l):
    for j in range(i-(window),i-1):
        if l[i]!=l[j]:
            return False
    return True
for i in range(window-1,len(l)):  
    if check_prev(i,window,l):
        state=l[i]    
        if state== previous_state or state=="None":
            continue
        else:
            dq.append(state)
            
        if len(dq)==3 and dq[0]=="standing" and dq[2]=="standing"and dq[1]=="squats":
            count+=1
        previous_state=state
print(count)

            
        