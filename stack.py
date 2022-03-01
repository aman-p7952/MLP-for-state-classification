#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:58:01 2022

@author: aman
"""

import os
import glob
import pandas as pd

filelist = ["squats/squats0.csv",
            "squats/squats1.csv",
            "squats/squats2.csv",
            "squats/squats3.csv",
            "squats/squats4.csv",
            "squats/squats5.csv",
            "squats/squats6.csv",
            "squats/squats7.csv",
            "squats/squats8.csv",
            "squats/squats9.csv",
            "pushup/push1.csv",
            "pushup/push2.csv",
            "pushup/push3.csv",
            "pushup/push4.csv",
            # "pushup/push5.csv",
            "pushup/push6.csv",
            "pushup/push7.csv",
            "pushup/push8.csv",
            "pushup/push9.csv",
            'pushup/push10.csv',
            'pushup/push11.csv',
            'pushup/push12.csv',
            "pushup/push13.csv",
            "pushup/push14.csv",
            #"pushup/push15.csv",
            "highknees/highknees0.csv",
            
            ]

dflist = []
for filename in filelist:
    df = pd.read_csv(filename,)
    dflist.append(df)
concatDf = pd.concat(dflist,axis = 0)
concatDf.to_csv("data.csv",index=None)