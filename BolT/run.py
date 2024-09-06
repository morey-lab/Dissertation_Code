from tqdm import tqdm
import torch
import numpy as np
import random
import os
import sys

from datetime import datetime

"""
function:
    Changes the current working directory to get the "utils" package and import it. Subsequent commands loads Option 
    function from the "utils" package

inputs:

outputs:


"""

if(not "utils" in os.getcwd()):
    sys.path.append("../../../")


from utils import Option
from utils import Option, calculateMetric

from model import Model
from Dataset.dataset import getDataset


"""
function:
    train(model, dataset, fold, nOfEpochs) trains a transformer model using the BolT architecture. It pulls the data
    in from a data fold and trains based on the number of epochs. It pulls a pytorch dataloaded training dataset using 
    the functionality of the dataset.py file. Each training set contains a set of subject IDs, subject labels, and BOLD
    time series corresponding to the individual subjects. Additionally, the training data has been split into k-training
    folds and a set of test folds for cross validation
    
    The tqdm() function causes any loop function to display a smart progress meter

inputs:
    model - specified ML model (transformer) to apply to the inputted dataset
    dataset - dataset to which the model is trained on (this is a fold (sample) of the data)
    fold - the fold ID used to identify the dataset fold
    nOfEpochs - Number of epochs used when training the ML model applied to data

outputs:
    preds - the predicted labels of the training data from the ML model
    probs - the probability associated with each label for each sample of the ML model
    groundTruths - the true value of the labels of each subject used to train
    losses - a loss value computed by the getLoss() function from models.py

"""

def train(model, dataset, fold, nOfEpochs):

    dataLoader = dataset.getFold(fold, train=True)

    for epoch in range(nOfEpochs):

            preds = []
            probs = []
            groundTruths = []
            losses = []

            for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
                
                xTrain = data["timeseries"] # (batchSize, N, dynamicLength)
                yTrain = data["label"] # (batchSize, )

                # NOTE: xTrain and yTrain are still on "cpu" at this point

                train_loss, train_preds, train_probs, yTrain = model.step(xTrain, yTrain, train=True)

                torch.cuda.empty_cache()

                preds.append(train_preds)
                probs.append(train_probs)
                groundTruths.append(yTrain)
                losses.append(train_loss)

            preds = torch.cat(preds, dim=0).numpy()
            probs = torch.cat(probs, dim=0).numpy()
            groundTruths = torch.cat(groundTruths, dim=0).numpy()
            losses = torch.tensor(losses).numpy()

            metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
            print("Train metrics : {}".format(metrics))                  


    return preds, probs, groundTruths, losses


"""
function:
    test(model, dataset, fold, nOfEpochs) tests a transformer model using the BolT architecture. It pulls the data
    in from a data fold and tests the model efficacy. It pulls a pytorch dataloaded test dataset using 
    the functionality of the dataset.py file. Each test set contains a set of subject IDs, subject labels, and BOLD
    time series corresponding to the individual subjects. Additionally, the test data has been split into k-test
    folds and a set of test folds for cross validation

    The tqdm() function causes any loop function to display a smart progress meter

inputs:
    model - specified ML model (transformer) to apply to the inputted dataset
    dataset - dataset to which the model is tested on (this is a fold (sample) of the data)
    fold - the fold ID used to identify the dataset fold

outputs:
    preds - the predicted labels of the test data from the ML model
    probs - the probability associated with each label for each sample of the ML model
    groundTruths - the true value of the labels of each subject used to test on
    losses - a loss value computed by the getLoss() function from models.py


"""

def test(model, dataset, fold):

    dataLoader = dataset.getFold(fold, train=False)

    preds = []
    probs = []
    groundTruths = []
    losses = []        

    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'Testing fold:{fold}')):

        xTest = data["timeseries"]
        yTest = data["label"]

        # NOTE: xTrain and yTrain are still on "cpu" at this point

        test_loss, test_preds, test_probs, yTest = model.step(xTest, yTest, train=False)
        
        torch.cuda.empty_cache()

        preds.append(test_preds)
        probs.append(test_probs)
        groundTruths.append(yTest)
        losses.append(test_loss)

    preds = torch.cat(preds, dim=0).numpy()
    probs = torch.cat(probs, dim=0).numpy()
    groundTruths = torch.cat(groundTruths, dim=0).numpy()
    loss = torch.tensor(losses).numpy().mean()          

    metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
    print("\n \n Test metrics : {}".format(metrics))                
    
    return preds, probs, groundTruths, loss
    

"""
function:
    run_bolT(hyperParams, datasetDetails, device="cuda:3", analysis=False) runs a sequence of previously
    defined functions that execute the data processing and training/test of the BolT transformer steps.
    Optionally, an analysis of the transformer model can be exported.
    
    For each fold generated a BolT transformer is trained and tested on the respective training and testing 
    datasets associated with that fold ID. Saves the performance of each model into the Results object and 

inputs:
    hyperParams - a dictionary that holds all the hyperparameters necessary for the BolT transformer model
    datasetDetails - dictionary object that holds metadata and parameters around each dataset used
    device - hardware selected by user to run the code off of
    analysis - True/False to generate a report holding model architecture information to a file path

outputs:
    results - saves a folds (both train and test) model performance to a list object


"""

def run_bolT(hyperParams, datasetDetails, device="cuda:3", analysis=False):


    # extract datasetDetails

    foldCount = datasetDetails.foldCount
    datasetSeed = datasetDetails.datasetSeed
    nOfEpochs = datasetDetails.nOfEpochs


    dataset = getDataset(datasetDetails)


    details = Option({
        "device" : device,
        "nOfTrains" : dataset.get_nOfTrains_perFold(),
        "nOfClasses" : datasetDetails.nOfClasses,
        "batchSize" : datasetDetails.batchSize,
        "nOfEpochs" : nOfEpochs
    })


    results = []

    for fold in range(foldCount):

        model = Model(hyperParams, details)


        train_preds, train_probs, train_groundTruths, train_loss = train(model, dataset, fold, nOfEpochs)   
        test_preds, test_probs, test_groundTruths, test_loss = test(model, dataset, fold)

        result = {

            "train" : {
                "labels" : train_groundTruths,
                "predictions" : train_preds,
                "probs" : train_probs,
                "loss" : train_loss
            },

            "test" : {
                "labels" : test_groundTruths,
                "predictions" : test_preds,
                "probs" : test_probs,
                "loss" : test_loss
            }

        }

        results.append(result)


        if(analysis):
            targetSaveDir = "./Analysis/TargetSavedModels/{}/seed_{}/".format(datasetDetails.datasetName, datasetSeed)
            os.makedirs(targetSaveDir, exist_ok=True)
            torch.save(model, targetSaveDir + "/model_{}.save".format(fold))


    return results
