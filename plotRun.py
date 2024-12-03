
#James Oswald 1/12/21

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import matplotlib.ticker as mtick


#takes a file path and returns a matplot figure object
def genFigure(filePath):
    data = np.load(filePath)
    countOfClustersTrain = dict(pd.DataFrame({'Label':data['trainLabel'].reshape(-1)})['Label'].value_counts().sort_index())
    countOfClustersVal = dict(pd.DataFrame({'Label':data['validLabel'].reshape(-1)})['Label'].value_counts().sort_index())
    _ =[countOfClustersVal.update({i:0}) for i in countOfClustersTrain.keys() if i not in countOfClustersVal.keys()]
    countOfClustersTrainFilter = list(countOfClustersTrain.values())
    countOfClustersValFilter = list(zip(*sorted(countOfClustersVal.items())))[1]
            
    beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo = data["par"]

    trainMat = data["tl"]
    epochsT = trainMat[:,0]
    rLossT = trainMat[:,1]
    kdLossT = trainMat[:,2]
    regLossT = trainMat[:,3]
    lossT = trainMat[:,4]
    accT = trainMat[:,5]
    scale = 100
    #validation losses vs epoch
    validMat = data["vl"]
    epochsV = validMat[:,0]
    rLossV = validMat[:,1]
    kdLossV = validMat[:,2]
    regLossV = validMat[:,3]
    lossV = validMat[:,4]
    accV = validMat[:,5]

    #corelations with each dimention vs epoch
    distence = data["distence"]
    distence_m = data["distence_m"]
    correlation = data["correlation"]
    correlation_valid = data["correlation_valid"]

    #fig, ax = plt.subplots(3)
    fig, ax = plt.subplots(6)
    fig.set_size_inches(8, 7)

    # fig2, ax2 = plt.subplots(math.ceil(len(distence[0])/3),3)
    # fig2.set_size_inches(8, 7)
    # fig3, ax3 = plt.subplots(math.ceil(len(distence_m[0])/3),3)
    # fig3.set_size_inches(8, 7)
    fig.suptitle( r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo) + " linearWidth" , fontsize=12)
    # fig2.suptitle( r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo) + " linearWidth" , fontsize=12)
    # fig3.suptitle( r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo) + " linearWidth" , fontsize=12)


    ax[0].plot(epochsT, rLossT, label="r Loss")
    ax[0].plot(epochsT, kdLossT, label="kld Loss")
    ax[0].plot(epochsT, regLossT, label="reg Loss")
    ax[0].plot(epochsT, lossT, label="Training Loss")
    ax[0].plot(epochsV, lossV, label="Valid Loss", ls="--")
    ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[0].set(xlabel="epoch", ylabel="T Loss")
    ax[0].label_outer()

    ax[1].plot(epochsT, rLossV, label="r Loss")
    ax[1].plot(epochsT, kdLossV, label="kld Loss")
    ax[1].plot(epochsT, regLossV, label="reg Loss")
    ax[1].plot(epochsV, lossV, label="Valid Loss")
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1].set(xlabel="epoch", ylabel="Val Loss")
    ax[1].label_outer()

    ax[2].plot(epochsT, accT, label="Train Accuracy")
    ax[2].plot(epochsV, accV, label="Valid Accuracy")
    ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[2].set(xlabel="epoch",ylabel="Accuracy")
    ax[2].text(x=1300, y=0.62, s=f"Training Accuracy: {'{:.3f}'.format(accT[len(accT) - 1])}")
    ax[2].text(x=1300, y=0.50, s=f"Testing Accuracy: {'{:.3f}'.format(accV[len(accV) - 1])}")
    ax[2].label_outer()

    ax[3].plot(list(range(0,len(distence)*scale,scale)),np.average(distence/countOfClustersTrainFilter,axis=1,weights=countOfClustersTrainFilter), label="Train Distance")
    ax[3].plot(list(range(0,len(distence_m)*scale,scale)),np.average(distence_m/countOfClustersValFilter,axis=1,weights=countOfClustersValFilter), label="Valid Distance")
    ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[3].set(xlabel="epoch",ylabel="WCD AVG")
    # ax[3].xticks(list(range(0,len(distence_m))))
    # ax[3].xticklabels(list(np.array(list(range(0,len(distence_m))))*10))
    # ax[3].text(x=1300, y=0.62, s=f"Training Distance: {'{:.3f}'.format(accT[len(accT) - 1])}")
    # ax[3].text(x=1300, y=0.50, s=f"Testing Distance: {'{:.3f}'.format(accV[len(accV) - 1])}")
    ax[3].label_outer()
    #for clusters in range(len(correlation[0])):
    ax[4].plot(list(range(0,len(correlation)*scale,scale)),correlation, label=f"Train correlation")
        #ax[4].plot(list(range(0,len(correlation_valid[:,clusters])*100,100)),correlation_valid[:,clusters], label=f"Valid correlation {clusters}")
    ax[4].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[4].set(xlabel="epoch",ylabel="T Corr")
    ax[4].label_outer()

    #for clusters in range(len(correlation[0])):
    ax[5].plot(list(range(0,len(correlation_valid)*scale,scale)),correlation_valid, label=f"Valid correlation")
    ax[5].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[5].set(xlabel="epoch",ylabel="Val Corr")
    ax[5].label_outer()


    plt.xticks(range(len(trainMat)))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    # clusterscolor=['R-FR','G', 'N', 'G-R', 'G-N', 'G-FR']
    # for di in range(len(distence[0])):
    #     y1 = np.array(distence[:,di])
    #     idx = di // 3
    #     jdx = di % 3
    #     #ax2[idx][jdx].plot(y1 , label=f'cluster {di}')
    #     ax2[idx][jdx].plot(y1 , label=clusterscolor[di])
    #     ax2[idx][jdx].legend()


    
    # for di_m in range(len(distence_m[0])):
    #     y1_ = np.array(distence_m[:,di_m])
    #     idx = di_m // 3
    #     jdx = di_m % 3
    #     #ax3[idx][jdx].plot(y1_ , label=f'cluster {di_m}')
    #     ax3[idx][jdx].plot(y1_ , label=clusterscolor[di_m])
    #     ax3[idx][jdx].legend()
    
    # ax[8].set(xlabel="epoch",ylabel='distence: '+str(round(np.mean(wavg),4)))
    # ax[8].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
    # ax[3].label_outer()
   

    fig.subplots_adjust(right=0.75)
    # fig2.subplots_adjust(right=0.75)
    # fig3.subplots_adjust(right=0.75)
    plt.setp(ax, xticks=list(range(0,len(trainMat),500)), xticklabels=list(range(0,len(trainMat),500)))    
    # plt.setp(ax2, xticks=list(range(0,len(distence_m))), xticklabels=list(np.array(list(range(0,len(distence_m))))*10))    
    # plt.setp(ax3, xticks=list(range(0,len(distence_m))), xticklabels=list(np.array(list(range(0,len(distence_m))))*10)) 
    # fig2.tight_layout()
    # fig3.tight_layout()
    return fig#,fig2,fig3
 



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="i", default="None")
parser.add_argument("-o", "--output", dest="o", default="None")
args = parser.parse_args()

""" if args.i != "None":
    f = genFigure(args.i)
    name = ""
    if args.o == "None":
        raise Exception("No output name provided")
        #name = "./graphs/b"+args.i+".png"
    else:
        name = args.o
    f.savefig(name)
else:
    fileName = "./runs/1623311492.npz"
    f = genFigure(fileName)
    f.savefig("./graphs/latest.png") """

def genPlotForRun(runsPath, run, graphsPath, graph):
    f = genFigure(runsPath+"/"+run)
    f.savefig(graphsPath+"/"+graph)
    # f2.savefig(graphsPath+"/2"+graph)
    # f3.savefig(graphsPath+"/3"+graph)


#genFigure("all-results/1-18-22-res/runs/a19lds19b0.007g1.0d1.0h13.npz")