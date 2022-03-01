#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:49:06 2022

@author: aman
"""
import torch
import mediapipe as mp
import numpy as np
import torch.nn as nn
HIDDEN_SIZE=100
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.main = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(132, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 7),
            #nn.Softmax()
        )
        
    def forward(self, x):
        return self.main(x)