
# James Oswald 1/12/20

#This file generates logs of runs with various parameters

import os
import time
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sequenceModel import SequenceModel
from sequenceTrainer import SequenceTrainer
from sequenceDataset import SequenceDataset
from plotRun import genPlotForRun

#CPU or GPU?
os.environ["CUDA_VISIBLE_DEVICES"]=""

batchSize = 32
numEpochs = 20
betas = [0.007]
gammas = [0]#, 0.8, 1, 1.3]
deltas = [0.1]#, 0.8, 2]
dropout =  [0]
latentDims= [20]
lstmLayers = [1]
hiddenSize = [10] #features inside lstm
trainTestSplit = (0.8, 0.2)
weighted = False


#Post processing of log data, avarages metrics for each epoch
def averageCols(logMat):
    # print("logMat")
    print(logMat.shape)
    rv = np.zeros((numEpochs, 6))
    for epoch in range(numEpochs):
        for col in range(1, 6):
            num = 0
            for row in range(logMat.shape[0]):
                if(logMat[row, 0] == epoch):
                    rv[epoch, col] += logMat[row, col]
                    num += 1
            rv[epoch, col] /= num
        rv[epoch,0] = epoch
    # print("rv", rv)
    return rv

paramDict = {"beta": betas, "gamma": gammas, "delta": deltas, 
     "latentDims": latentDims, "lstmLayers": lstmLayers, "dropout":dropout, "hiddenSize":hiddenSize}
for params in list(ParameterGrid(paramDict)):   #gridsearch
    #set up the model and trainer
    model = SequenceModel(hidden_layers=params["lstmLayers"], emb_dim=params["latentDims"], dropout=params["dropout"], hidden_size=params["hiddenSize"]) 
    data = SequenceDataset(split=trainTestSplit)
    if torch.cuda.is_available(): 
        print('cuda available')
        model.cuda()
    trainer = SequenceTrainer(data, model, beta=params["beta"], gamma=params["gamma"], delta=params["delta"], logTerms=True, IICorVsEpoch=True)
    if torch.cuda.is_available(): 
        trainer.cuda()
    
    #train model, use internal logging
    print("Training Model")
    # trainer.train_model(batchSize, numEpochs, log=False)

    #train model, using weighted loss function
    distence ,distence_m,correlation,correlation_valid = trainer.train_model(batchSize, numEpochs, log=False, weightedLoss=weighted)

    filename = "a" + str(params["latentDims"]) + "lds"+str(params["latentDims"])+"b"+str(params["beta"])+"g" +str(params["gamma"])+"d"+str(params["delta"])+"h"+str(params["hiddenSize"])
    #save the model
    if weighted:
        torch.save(model, "./models/weighted/" + filename + ".pt")
    else:
        torch.save(model, "./models/" + filename + ".pt")
    print("Avging stats for batches")
    #training accuricies at each epoch
    tl = averageCols(trainer.trainList) 
    #validation accuricies at each epoch
    vl = averageCols(trainer.validList)

    

    print("Saving file")
    par = np.array([ params["beta"], params["gamma"], params["delta"], 
        params["latentDims"], params["lstmLayers"], params["dropout"], params["hiddenSize"]])
    if weighted:
        np.savez("./runs/weighted/" + filename + ".npz", par=par, tl=tl, vl=vl, distence=distence,distence_m=distence_m,correlation=correlation,correlation_valid=correlation_valid,trainLabel = data.train_label,validLabel = data.val_label)
    else:
        np.savez("./runs/" + filename + ".npz", par=par, tl=tl, vl=vl, distence=distence,distence_m=distence_m,correlation=correlation,correlation_valid=correlation_valid,trainLabel = data.train_label,validLabel = data.val_label)

    if weighted:
        genPlotForRun(runsPath="./runs/weighted/", run=filename + ".npz", graphsPath="./graphs/weighted", graph=filename + ".png")
    else:
        genPlotForRun(runsPath="./runs/", run=filename + ".npz", graphsPath="./graphs", graph=filename + ".png")
    
