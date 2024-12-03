#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:07:00 2020

@author: alexandergorovits
"""
import os
import pandas as pd
import numpy as np
from scipy.sparse.sputils import validateaxis
import torch
from torch.utils.data import DataLoader, TensorDataset , WeightedRandomSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import sys
os.chdir(sys.path[0]) #compatability hack


def dataSampler(labels):
    

    class_sample_count  = np.unique(labels, return_counts=True)[1]
    weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
    samples_weights = weights[labels]
    samples_weights = torch.reshape(samples_weights, (-1,))
    return WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

# class RegressionDataset(Dataset):
#     def __init__(self, X_data, indices=None):
#     # def __init__(self, X_data, adjs, indices=None):
#         self.X_data = X_data
#         # self.adjs = adjs
#         self.indices = indices

#     def __getitem__(self, index):
#         # print("index    ", index)
#         return self.X_data[index], index

#     def __len__(self):
#         return len(self.X_data)


class SequenceDataset:
    
    def __init__(self,datafile = './data-and-cleaning/GMMdata.xlsx', seqlen=10, split=(0.85, 0.15), noofbuckets = 7):
        if datafile[-4:] == 'xlsx':
            self.dataset = pd.read_excel(datafile)
        else:
            self.dataset = pd.read_csv(datafile)
        if "Unnamed: 0" in self.dataset.columns:
            self.dataset.drop(columns=["Unnamed: 0"],inplace=True)
        self.ALPHABET = ['A','C','G','T']
        self.seqlen = seqlen
        self.split = split
        self.noofbuckets = noofbuckets
        
    def transform_sequences(self,seqs):
        enc = OneHotEncoder()
        enc.fit(np.array(self.ALPHABET).reshape(-1,1))
        return enc.transform(seqs.reshape(-1,1)).toarray().reshape(
            -1, self.seqlen, len(self.ALPHABET))
        
    # @staticmethod
    # def cust_collate(batch):
    #     #print(list(batch))
    #     #print(len(batch))
    #     return [x for x in batch]
    # TODO: normalize a datas normlize by the sum    
    def data_loaders(self, batch_size):
        seqs = self.transform_sequences(
            self.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).to_numpy())
        Dlables = self.dataset['Label'].to_numpy(dtype="float").reshape(-1,1)
        
        
        nval = seqs.shape[0]
        split1 = int(self.split[0]*nval)
        split2 = int(self.split[1]*nval)
        perm = torch.randperm(nval)
        if self.split[1] != 0:
            self.train_seq , self.val_seq , self.train_label , self.val_label = train_test_split(seqs,Dlables,test_size=self.split[1],random_state=42)
        else:
            self.train_seq = seqs
            self.val_seq = np.zeros_like(seqs)
            self.train_label = Dlables
            self.val_label = np.zeros_like(Dlables)

        print(nval, split1, split2)
        
        train_ds = TensorDataset(
            torch.from_numpy(self.train_seq),
            torch.from_numpy(self.train_label)
        )

        val_ds = TensorDataset(
            torch.from_numpy(self.val_seq),
            torch.from_numpy(self.val_label)
        )

        
        print(len(train_ds),len(val_ds))
        
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False
            )
        
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False
            )
        
        # print("inja0", len(train_dl[0]),len(val_dl))
        print("inja1", type(train_dl))
        print("inja2", np.shape(val_dl))
        
        
        return train_dl, val_dl, None

# aa = SequenceDataset()
# traindl ,val_dl ,_ , _ = aa.data_loaders(32)
# for i in traindl:
#     print(i)