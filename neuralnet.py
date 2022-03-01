#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 17:49:21 2022
@author: aman
"""
import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from net import Net1
df = pd.read_csv("./data.csv")
df = df.drop("frame", axis=1)
# df = df[df.state != "None"]
df.index = np.arange(0, len(df))
print(df)
print(df["state"].value_counts())
num_classes = 7
# key = {"standing": 0,
#        "squats": 1,
#        "None":2}
key = {"pushup up": 0,
        "pushup down": 1,
        "None":2,
        "nan":2,
        "standing": 3,
        "squats": 4,
        "left knee high":5,
        "right knee high":6,}
        

class PoseDataset(Dataset):
    def __init__(self, file):
        self.file = file
        df = pd.read_csv(file)
        
        self.df = df.drop("frame", 1)
        self.df.dropna(axis=0,how="any",inplace=True)
        # self.df = df[df.state != "None"]
        self.df.index = np.arange(0, len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature = self.df.iloc[idx, 0:132]
        feature = pd.to_numeric(feature, errors='coerce')
        label = self.df.iloc[idx, 132]
        # print(torch.tensor(feature).type)
        sample = {"feature": torch.tensor(feature),"label":torch.tensor(key[label])}
        
        return sample
net = Net1()
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = PoseDataset("./data.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    # dataset_test = PoseDataset(root_path_test)
    dataloader = DataLoader(train_dataset, batch_size=100,
                            shuffle=True, num_workers=0, drop_last=True)
    dataloader_test = DataLoader(test_dataset, batch_size=100, drop_last=True)
    num_epochs = 25
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    critarian = nn.CrossEntropyLoss()
    train_history = []
    test_history = []
    losses = []
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            features = batch["feature"].float()
            label = batch["label"].long()
            output = net(features)
            loss = critarian(output, label)
            loss.backward()
            optimizer.step()
            # print(output)
            losses.append(loss.item())
        with torch.no_grad():
            correct_train = 0
            total_train = 0
            correct_test = 0
            total_test = 0
            for data in dataloader:
                features = data["feature"].float()
                label = data["label"].float()
                outputs = net(features)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                #_, label = torch.max(label.data, 1)
                total_train += label.size(0)
                correct_train += (predicted == label).sum().item()

            train_acc = correct_train/total_train
            train_history.append(train_acc)

            for data in dataloader:
                features = data["feature"].float()
                label = data["label"].float()
                outputs = net(features)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                #_, label = torch.max(label.data, 1)
                total_test += label.size(0)
                correct_test += (predicted == label).sum().item()
            test_acc = correct_test/total_test
            test_history.append(test_acc)
            # print('Accuracy of the network on the test images: %f ' % (
            #    correct_test / total_test))
            # print('Accuracy of the network on the train images: %f ' % (
          #   correct_train / total_train))
        loss_mean = np.array(losses[-50:]).mean()
        print("epoch: %d mean_loss %f train_accracy %f  test_accracy %f" %
              (epoch+1, loss_mean, train_acc, test_acc))
torch.save(net.state_dict(), "net.pt")
