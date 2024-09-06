
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from random import shuffle, randrange
import numpy as np
import random

#from hcpRestLoader import hcpRestLoader
#from hcpTaskLoader import hcpTaskLoader
from abide1Loader import abide1Loader

# Create dictionary that associates a dataset with a load .py file for that dataset
loaderMapper = {
    #"hcpRest" : hcpRestLoader,
    #"hcpTask" : hcpTaskLoader,
    "abide1" : abide1Loader,
}

"""
function:
    getDataset(options) generates the SupervisedDataset() class object. It passes the options values to this class to 
    generate the SupervisedDataset() object (see output below)

inputs:

outputs:

"""


def getDataset(options):
    return SupervisedDataset(options)


"""
function:
    SupervisedDataset(Dataset) creates an objects that contain the functions: __init__(), __len__(),
    get_nOfTrains_perFold(), setFold(), etc.
    
inputs:
    Dataset - tensor object that contains the BOLD signals or any similar data
    
outputs:
    Dataset object (tensor) with built-in functions (see function below)
"""

class SupervisedDataset(Dataset):

    """
    function:
        __init__(self, datasetDetails) builds in metadata information from the datasetDetails imported.
        Builds in a function that generates k stratified K-folds based on the number of splits given by
        datasetDetails.foldCount

    inputs:
        datasetDetails - .py file that holds metadata information and contains some hyperparameter information
                         for each dataset

    outputs:
        Embeds the information produced by __init__() function into the inputted dataset object
    """
    
    def __init__(self, datasetDetails):

        self.batchSize = datasetDetails.batchSize
        self.dynamicLength = datasetDetails.dynamicLength
        self.foldCount = datasetDetails.foldCount

        self.seed = datasetDetails.datasetSeed

        loader = loaderMapper[datasetDetails.datasetName]

        self.kFold = StratifiedKFold(datasetDetails.foldCount, shuffle=False, random_state=None) if datasetDetails.foldCount is not None else None
        self.k = None

        self.data, self.labels, self.subjectIds = loader(datasetDetails.atlas, datasetDetails.targetTask)

        random.Random(self.seed).shuffle(self.data)
        random.Random(self.seed).shuffle(self.labels)
        random.Random(self.seed).shuffle(self.subjectIds)

        self.targetData = None
        self.targetLabel = None
        self.targetSubjIds = None

        self.randomRanges = None

        self.trainIdx = None
        self.testIdx = None

    """
    function:
        __len__(self) returns the length of dataset (No. of samples)

    inputs:

    outputs:
        
    """

    def __len__(self):
        return len(self.data) if isinstance(self.targetData, type(None)) else len(self.targetData)

    """
    function:
        get_nOfTrains_perFold(self) returns the length of each fold or the length of the dataset

    inputs:

    outputs:

    """
    def get_nOfTrains_perFold(self):
        if(self.foldCount != None):
            return int(np.ceil(len(self.data) * (self.foldCount - 1) / self.foldCount))
        else:
            return len(self.data)        

    """
    function:
        setFold(self, fold, train=True) grabs the number of folds from the data. It then splits the 
        data and associated labels of the data into training and test splits

    inputs:

    outputs:
        embeds training and test folds into the dataset
    """
    def setFold(self, fold, train=True):

        self.k = fold
        self.train = train


        if(self.foldCount == None): # if this is the case, train must be True
            trainIdx = list(range(len(self.data)))
        else:
            trainIdx, testIdx = list(self.kFold.split(self.data, self.labels))[fold]      

        self.trainIdx = trainIdx
        self.testIdx = testIdx

        random.Random(self.seed).shuffle(trainIdx)

        self.targetData = [self.data[idx] for idx in trainIdx] if train else [self.data[idx] for idx in testIdx]
        self.targetLabels = [self.labels[idx] for idx in trainIdx] if train else [self.labels[idx] for idx in testIdx]
        self.targetSubjIds = [self.subjectIds[idx] for idx in trainIdx] if train else [self.subjectIds[idx] for idx in testIdx]

        if(train and not isinstance(self.dynamicLength, type(None))):
            np.random.seed(self.seed+1)
            self.randomRanges = [[np.random.randint(0, self.data[idx].shape[-1] - self.dynamicLength) for k in range(9999)] for idx in trainIdx]

    """
    function:
        getFold(self, fold, train=True) examines given fold look to see if it a training or test.
        If it is a training set it is passed to the pytorch Dataloader function which combines 
        a dataset and a sampler, and provides an iterable over the given dataset. If the data passed
        is a test dataset then is passes it to the Dataloader but with different batch size. the train 
        value need to declared "False" for test splits

    inputs:
        
    outputs:
        Pytorch Dataloader object with training and test sets
    """

    def getFold(self, fold, train=True):
        
        self.setFold(fold, train)

        if(train):
            return DataLoader(self, batch_size=self.batchSize, shuffle=False)
        else:
            return DataLoader(self, batch_size=1, shuffle=False)

    """
    function:
        __getitem__(self, idx) based on a fold index value the user can extract the properties of this dataset.
        This includes the BOLD series, subject labels and subject IDs, etc.

    inputs:

    outputs:
        Embeds the __getitem__() function into the passed dataset
    """

    def __getitem__(self, idx):
        
        subject = self.targetData[idx]
        label = self.targetLabels[idx]
        subjId = self.targetSubjIds[idx]


        # normalize timeseries
        timeseries = subject # (numberOfRois, time)

        timeseries = (timeseries - np.mean(timeseries, axis=1, keepdims=True)) / np.std(timeseries, axis=1, keepdims=True)
        timeseries = np.nan_to_num(timeseries, 0)

        # dynamic sampling if train
        if(self.train and not isinstance(self.dynamicLength, type(None))):
            if(timeseries.shape[1] < self.dynamicLength):
                print(timeseries.shape[1], self.dynamicLength)

            samplingInit = self.randomRanges[idx].pop()

            timeseries = timeseries[:, samplingInit : samplingInit + self.dynamicLength]

        return {"timeseries" : timeseries.astype(np.float32), "label" : label, "subjId" : subjId}







