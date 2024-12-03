#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:32:05 2020

@author: alexandergorovits
"""
#import os
#import json
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch.nn as nn
from utils.trainer import Trainer
#from sequenceModel import SequenceModel
from utils.helpers import to_cuda_variable_long, to_cuda_variable, to_numpy
#from utils.evaluation import *

#The Inherited Trainer Class
class SequenceTrainer(Trainer):
    REG_TYPE = {"integrated_intensity": 0}
    def __init__(
            self,
            dataset,
            model,
            lr=1e-4,
            reg_type = 'all',
            reg_dim = tuple([0]),
            beta=0.001,
            gamma=1.0,
            capacity=0.0,
            rand=0,
            delta=10.0,
            logTerms=False, #James's NPZ logging system
            IICorVsEpoch=False, #Flag to disable faulty cuda IICorVsEpoch Logging when running on GPU
            alpha = 5.0
    ):
        super(SequenceTrainer, self).__init__(dataset, model, lr)
        
        self.attr_dict = self.REG_TYPE  
        self.reverse_attr_dict = {  #map of regularized dimension indices to their names
            v: k for k, v in self.attr_dict.items()
        }
        self.metrics = {}
        self.beta = beta    #The loss hyperparameters, 
        self.gamma = 0.0    # g and d are reset later in this constructor, update them there not here
        self.delta = 0.0
        self.capacity = to_cuda_variable(torch.FloatTensor([capacity])) 
        self.cur_epoch_num = 0      #The current Epoch while training
        self.warm_up_epochs = 10    #This dosn't do anything anywhere? CTRL+F
        self.reg_type = reg_type    #configures regularization settings later on
        self.reg_dim = ()           #The dimentions we're looking to regularize
        self.use_reg_loss = False   #use regularization loss, without this its just Beta VAE
        self.rand_seed = rand
        self.logTerms = logTerms 
        if logTerms: #use James's logging model
            self.trainList = np.zeros((0,6)) #Training accuracy after each epoch
            self.validList = np.zeros((0,6)) #validation accuracy after each epoch
            self.IICorVsEpoch = IICorVsEpoch                   #Do we log II vs epoch or no?
            if IICorVsEpoch:
                self.WLCorList = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch
                self.LIICorList = np.zeros((0, self.model.emb_dim)) #II corelation of all dims after each epoch
        
        torch.manual_seed(self.rand_seed)
        np.random.seed(self.rand_seed)
        self.trainer_config = f'_r_{self.rand_seed}_b_{self.beta}_'
        if capacity != 0.0:
            self.trainer_config += f'c_{capacity}_'
        self.model.update_trainer_config(self.trainer_config)
        if len(self.reg_type) != 0: # meaning we're using an ARVAE, not just beta VAE
            self.use_reg_loss = True
            self.reg_dim = reg_dim
            self.gamma = gamma
            self.delta = delta
            self.alpha = alpha
            self.trainer_config += f'g_{self.gamma}_d_{self.delta}_'
            reg_type_str = '_'.join(self.reg_type)
            self.trainer_config += f'{reg_type_str}_'
            self.model.update_trainer_config(self.trainer_config)

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        #score tensor is actually one hot encodings
        score_tensor, attribTesnsor = batch
        # convert input to torch Variables
        batch_data = (
            to_cuda_variable_long(score_tensor),
            to_cuda_variable_long(attribTesnsor)
        )
        return batch_data

    
    


    #This is our primary loss function 
    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True, weightedLoss=False, probBins=[]):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param batch_num: int
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        if self.cur_epoch_num != epoch_num:
            flag = True
            self.cur_epoch_num = epoch_num
        else:
            flag = False

        inputs, labels = batch
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs, latent_dist, prior, latent_sample, _, mean_z, std_z , property_prediction = self.model(inputs)
        
        # compute accuracy
        accuracy = self.mean_accuracy(weights=outputs,target=inputs)
        r_loss = self.reconstruction_loss(inputs, outputs)
        kld_loss = self.compute_kld_loss(latent_dist, prior, beta=self.beta, c=self.capacity)
        analyzer_loss = nn.functional.mse_loss(property_prediction.squeeze(), labels.float())
        loss = r_loss + kld_loss + analyzer_loss
        
        #smmplDistence = self.Distencecs(labels)
        # compute and add regularization loss if needed, otherwise its just beta-VAE
        if self.use_reg_loss:
            reg_loss = 0.0
            if type(self.reg_dim) == tuple:
                #for dim in self.reg_dim:
                    #our supervized training
                if weightedLoss == False:
                    reg_loss += self.compute_reg_loss(
                        latent_sample,inputs,labels, mean_z, std_z, gamma=self.gamma ,train = train, factor=self.delta )
                else:
                    reg_loss += self.compute_reg_loss_weighted(
                    self, latent_sample,inputs, labels 
                    , gamma=self.gamma, alpha = self.alpha,
                    factor=self.delta,
                    probBins=probBins
                    )
            else:
                raise TypeError("Regularization dimension must be a tuple of integers")
            loss += reg_loss

            #James's "worry about it later by throwing it all into a matrix" logging system for loss terms
            if self.logTerms and train:
                self.trainList = np.vstack((self.trainList, 
                    [self.cur_epoch_num, r_loss.item(), kld_loss.item(),
                    reg_loss.item(), loss.item(), accuracy.item()]))
            if self.logTerms and not train:
                self.validList = np.vstack((self.validList, 
                    [self.cur_epoch_num, r_loss.item(), kld_loss.item(),
                    reg_loss.item(), loss.item(), accuracy.item()]))

        return loss, accuracy

    def compute_representations(self, data_loader, num_batches=None):
        latent_codes = []
        attributes = []
        if num_batches is None:
            num_batches = 200
        for batch_id, batch in tqdm(enumerate(data_loader)):
            inputs, metadata = self.process_batch_data(batch)
            _, _, _, z_tilde, _ = self.model(inputs)
            latent_codes.append(to_numpy(z_tilde.cpu()))
            labels = metadata
            attributes.append(to_numpy(labels))
            if batch_id == num_batches:
                break
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)
        attr_list = [
            attr for attr in self.attr_dict.keys()
        ]
        return latent_codes, attributes, attr_list

    #This function is called after each epoch, 
    #we can use it for computing metrics of the model after each epoch.
    
    def loss_and_acc_test(self, data_loader):
        mean_loss = 0
        mean_accuracy = 0

        for _, batch in tqdm(enumerate(data_loader)):
            inputs, _ = self.process_batch_data(batch)
            inputs = to_cuda_variable(inputs)
            # compute forward pass
            outputs, _, _, _, _ = self.model(inputs)
            # compute loss
            recons_loss = self.reconstruction_loss(
                x=inputs, x_recons=outputs
            )
            loss = recons_loss
            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            accuracy = self.mean_accuracy(
                weights=outputs,
                target=inputs
            )
            mean_accuracy += to_numpy(accuracy)
        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    @staticmethod
    def reconstruction_loss(x, x_recons):
        return Trainer.mean_crossentropy_loss(weights=x_recons, targets=x.argmax(dim=2))

    @staticmethod
    def mean_accuracy(weights, target):
        _,_,nn = weights.size()
        weights = weights.view(-1,nn)
        target = target.argmax(dim=2).view(-1)

        _, best = weights.max(1)
        correct = (best==target)
        return torch.sum(correct.float())/target.size(0) 
