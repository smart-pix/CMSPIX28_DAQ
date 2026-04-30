#Author: Daniel Abadjiev, based on run.ipynb by Danush Sekhar/Anthony Badea, see also https://github.com/smart-pix/filter
#Date: September 2025, moved to lab_analysis folder October 2025
#Description: Run on yprofiles from ASIC and generate qkeras predictions for them, 
#and then make a confusion matrix out of that
##############################################
#easiest usage is with bashQkerasConfusionMatrix.sh bash script
#or, if you have a dataPath with final_results.npy and yprofiles.csv as saved
#by dynamic_parseDNNresults.py and ROUTINE_DNN respectively, you can 
#run this script with
#python just_qkeras_predict.py -d dataPath
##############################################



import os
import numpy as np
import tensorflow as tf
import csv
import pandas as pd
#add paths so these imports work
import sys
ucFilterGithubPath = "/local/d1/smartpixLab/filter/"
fermiFilterGithubPath = "/path/to/filter"
cornelFilterGithubPath = "/path/to/filter"
filterGithubPath = ucFilterGithubPath
sys.path.append(filterGithubPath)
sys.path.append(os.path.join(filterGithubPath,"model_pipeline"))
sys.path.append(os.path.join(filterGithubPath,"pretrain-data-prep"))
import model as md
import utils as ut
import argparse
import matplotlib.pyplot as plt

from qkeras import QDenseBatchnorm

#Now a default for where the qmodel file is 
#from fermilab's original code:
QMODEL_FILE_PATH = "/fasic_home/gdg/research/projects/CMS_PIX_28/directional-pixel-detectors/multiclassifier/models/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5"
#For Cornell 
QMODEL_FILE_PATH = "/path/to/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5" 
#For UC 
QMODEL_FILE_PATH = "/local/d1/smartpixLab/fermiModels/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5" 



def main():
    parser = argparse.ArgumentParser(description='Parse ASIC yprofile results and generate qkeras predictions')
    parser.add_argument("-d", "--dataPath", type=str, default='/local/d1/smartpixLab/scurveData/ChipVersion1_ChipID16_SuperPix2/2025.09.02_mergedRun/', help='ASIC data path')
    args = parser.parse_args()

    genQkerasPredictsForASICPath(args.dataPath)
    plotConfusionMatrix(args.dataPath)

def genQkerasPredictsForASICPath(dataPath):

    # dataPath = "/local/d1/smartpixLab/scurveData/ChipVersion1_ChipID16_SuperPix2/2025.09.02_13.43.34_DNN/"
    # dataPath = "/home/daq/smartpix/smartpixLab/scurveData/ChipVersion1_ChipID16_SuperPix2/2025.09.02_mergedRun"
    yprofilePath = os.path.join(dataPath,"yprofiles.csv")
    yprofiles = np.genfromtxt(yprofilePath, delimiter=',', dtype=int)


    confs = [
        # {"qm_charge_levels" : [400, 1600, 2400], "qm_quant_values" : [0, 1, 2, 3]},
        # {"qm_charge_levels" : [1000, 1600, 2400], "qm_quant_values" : [0, 1, 2, 3]},
        {"qm_charge_levels" : [923, 1847, 3695], "qm_quant_values" : [0, 1, 2, 3]},
    ]



    # import qkeras 
    # from qkeras import QDense
    # from qkeras import QConv1D
    # from qkeras import QBatchNormalization
    # from qkeras import QDenseBatchnorm
    # qkeras.QDenseBatchnorm();
    # QBatchNormalization.QDenseBatchnorm();

    # create model
    shape = 16 # y-profile ... why is this 16 and not 8?
    nb_classes = 3 # positive low pt, negative low pt, high pt
    first_dense = 58 # shape of first dense layer
    # qkeras model
    qmodel_file = "/fasic_home/gdg/research/projects/CMS_PIX_28/directional-pixel-detectors/multiclassifier/models/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5"
    qmodel_file = "/local/d1/smartpixLab/fermiModels/ds8l6_padded_noscaling_qkeras_foldbatchnorm_d58w4a8model.h5"
    qmodel = md.CreateQModel(shape, model_file=qmodel_file)

    # Compute loss and accuracy manually
    def getLA(y, predictions, loss_fn, acc_metric=tf.keras.metrics.SparseCategoricalAccuracy()):
        loss = loss_fn(y, predictions).numpy()
        acc_metric.update_state(y, predictions)
        accuracy = acc_metric.result().numpy()
        return loss, accuracy

    # evaluating
    verbose = 1
    batch_size = 2048

    # loop over the confs
    for conf in confs:
        
        # make predictions
        for m, name in zip([qmodel], ["qkeras"]):
            assert name =="qkeras"
            print(f"Evaluating {name} model...")
            conf[f"{name}_predictions"] = m.predict(yprofiles, batch_size = batch_size, verbose=verbose)
            conf["qkeras_predictions_argmax"] = np.argmax(conf["qkeras_predictions"], axis=1)
            # predictions = np.argmax(predictions, axis=1)
            outDir = os.path.join(dataPath, "_".join(map(str, conf["qm_charge_levels"])))
            os.makedirs(outDir,exist_ok=True)
            predFileName = os.path.join(outDir, f"{name}_predictions.npy")
            predFileName2 = os.path.join(outDir, f"{name}_predictions_argmax.npy")
            predFileName2_csv = os.path.join(outDir, f"{name}_predictions_argmax.csv")

            np.save(predFileName, conf[f"{name}_predictions"])
            np.save(predFileName2, conf[f"{name}_predictions_argmax"])
            (conf[f"{name}_predictions_argmax"]).tofile(predFileName2_csv,sep='\n')
            # model_loss, model_acc = getLA(conf["qm"]["clslabels"], conf[f"{name}_predictions"], md.custom_loss_function)
            # print(f"Finished evaluating {name} model with loss: {model_loss}, accuracy: {model_acc}, predictions saved to {predFileName}")
            print()

#returns the confusion matrix
def plotConfusionMatrix(dataPath,runTitle="",QkerasResults = None, ASICresults = None,ASICStrLabel = "ASIC", qkerasStrLabel = "Qkeras"):
    
    
    if ASICresults is None:
        ASICresults = np.load(os.path.join(dataPath,"final_results.npy"))
    if QkerasResults is None:
        QkerasResults = np.load(os.path.join(dataPath,"923_1847_3695","qkeras_predictions_argmax.npy"))
    assert len(QkerasResults)==len(ASICresults)

    if len(runTitle)==0:
        runTitle = dataPath[-56:-24] + "\n"+ dataPath[-24:] + f"\n{len(QkerasResults)} test vectors"

    #from Danush's code
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(ASICresults, QkerasResults, labels=[0, 1, 2])
    cm_totalNormalized = cm/len(QkerasResults)
    print("debug")
    #print(cm)
    #print(np.sum(cm,axis=1))
    #cm_colNormalized = cm/np.sum(cm,axis=1)  #this is actually row normalized
    cm_colNormalized = np.divide(cm,np.sum(cm,axis=0))
    assert len(QkerasResults) == np.sum(np.sum(cm))
    print(cm)
    print(f"total normalized:\n{cm_totalNormalized}")
    print(f"column normalized:\n{cm_colNormalized}")
    offAxis = cm[0,1]+cm[0,2]+cm[1,0]+cm[1,2]+cm[2,0]+cm[2,1]
    accuracy = (cm[0,0]+cm[1,1]+cm[2,2]) / len(QkerasResults)
    print(f"Bad/off-axis vectors: {offAxis}")
    print(f"Overall accuracy: {accuracy*100}%")

    #my plot of 2d hist
    plt.figure(figsize=(8,15))
    plt.subplot(311)
    plt.hist2d(QkerasResults,ASICresults,cmin=1,bins=[[0,1,2,3],[-1,0,1,2,3]])
    plt.xlabel(qkerasStrLabel+" predictions")    
    plt.ylabel(ASICStrLabel+" predictions")
    plt.title(runTitle)
    plt.colorbar()


    #Danush's way to plot this, slightly edited so it's all one big figure
    
    # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    plt.subplot(312)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0','1','2'], yticklabels=['0','1','2'])
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['High p$_T$', 'Low p$_T$ negative', 'Low p$_T$ positive'], yticklabels=['High p$_T$', 'Low p$_T$ negative', 'Low p$_T$ positive'])
    plt.ylabel(ASICStrLabel+' label')
    plt.xlabel(qkerasStrLabel+' label')
    plt.title('Confusion Matrix')
    plt.subplot(313)
    
    sns.heatmap(cm_colNormalized, annot=True,  cmap='Blues', xticklabels=['0','1','2'], yticklabels=['0','1','2'])
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['High p$_T$', 'Low p$_T$ negative', 'Low p$_T$ positive'], yticklabels=['High p$_T$', 'Low p$_T$ negative', 'Low p$_T$ positive'])
    plt.ylabel(ASICStrLabel+' label')
    plt.xlabel(qkerasStrLabel+' label')
    plt.title('Confusion Matrix Normalized by Column')
    plt.savefig(os.path.join(dataPath,'final_results_confusion_matrix.pdf'), dpi=300)
    plt.show()

    return cm, cm_totalNormalized,cm_colNormalized

    # sns.heatmap(cm/len(QkerasResults), annot=True, cmap='Blues', xticklabels=['0','1','2'], yticklabels=['0','1','2'])



if __name__ == "__main__":
    main()
