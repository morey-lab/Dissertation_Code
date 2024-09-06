from bolT import BolT
import torch
import numpy as np
from einops import rearrange


class Model:

    """
    function:
        __init__() embeds the hyperparameters for the BolT transformer model (number of embedding dimensions, number of
        attention heads, window sizes, window shift size, etc.) into an object. It also initializes the structure of the
        transformer model before it is trained

    inputs:
        hyperparameters - Initialized from a .py file that establishes the models hyperparameters
        details - python object that holds various pieces of parameter and meta-data information

    outputs:
        Embeds metadata, parameter information, and initial BolT model architecture into an object
    """

    def __init__(self, hyperParams, details):

        self.hyperParams = hyperParams
        self.details = details

        # Initialzes transformer model architecture using the hyperparameters and initialized weights
        self.model = BolT(hyperParams, details)

        # load model into gpu
        
        self.model = self.model.to(details.device)

        # set criterion (sets cross entropy function)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)#, weight = classWeights)
       
        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = hyperParams.lr, weight_decay = hyperParams.weightDecay)

        # set scheduler
        steps_per_epoch = int(np.ceil(details.nOfTrains / details.batchSize))        
        
        divFactor = hyperParams.maxLr / hyperParams.lr
        finalDivFactor = hyperParams.lr / hyperParams.minLr
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, hyperParams.maxLr, details.nOfEpochs * (steps_per_epoch), div_factor=divFactor, final_div_factor=finalDivFactor, pct_start=0.3)

    """
    function:
        step(self, x, y, train=True) creates a built-in function that loads the subject ROI data (x) and 
        subject label data (y) into the GPU (via the prepareInputs function)

    inputs:
        x - ROI BOLD signals embeddings
        y - label of subject corresponding to respective ROI BOLD signal data
        train - True/False option that determines if the data given is training or test data

    outputs:
        loss - outputs computed loss value of data given
        preds - predictions outputted by the BolT model given the inputted data
        probs - probabilities associated with each class outputted by the BolT model
        y - the true labels of the subject(s) for the given data
        
    """

    def step(self, x, y, train=True):

        """
            x = (batchSize, N, dynamicLength) 
            y = (batchSize, numberOfClasses)

        """

        # PREPARE INPUTS
        
        inputs, y = self.prepareInput(x, y)  # references prepareInput(x, y) function below

        # DEFAULT TRAIN ROUTINE
        
        if(train):
            self.model.train()
        else:
            self.model.eval()

        yHat, cls = self.model(*inputs)
        loss = self.getLoss(yHat, y, cls)

        preds = yHat.argmax(1)
        probs = yHat.softmax(1)

        if(train):

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if(not isinstance(self.scheduler, type(None))):
                self.scheduler.step()            

        loss = loss.detach().to("cpu")
        preds = preds.detach().to("cpu")
        probs = probs.detach().to("cpu")

        y = y.to("cpu")
        
        torch.cuda.empty_cache()


        return loss, preds, probs, y

    """
    function:
        prepareInput(self, x, y) loads the ROI BOLD signal embeddings into the GPU (or user chosen local device).
        The GPU (or other device) is where the model is learned. Eventually the final model is saved to some 
        folder location and the GPU is cleared

    inputs:
        x - ROI BOLD signals embeddings
        y - label of subject corresponding to respective ROI BOLD signal data
        train - True/False option that determines if the data given is training or test data

    outputs:
        (x, ) - 3-d tensor object inputted into the GPU (or other device)
        y - labels corresponding to each subject(s) used to train/test model

    """

    # HELPER FUNCTIONS HERE

    def prepareInput(self, x, y):

        """
            x = (batchSize, N, T): batch size = # of subjects, N = N-dimensional embedding, T = BOLD time points sampled
            y = (batchSize, )

        """
        # to gpu now

        x = x.to(self.details.device)
        y = y.to(self.details.device)


        return (x, ), y

    """
    function:
        getLoss(self, yHat, y, cls) examines the current BolT classification prediction(s) of a set of subjects
        along with the true classifications of the subjects. It computes a loss value based on cross entropy 
        (cross_entropy_loss) and mean of the squared difference between the CLS tokens and the mean of the CLS tokens 
        (CLS loss). 
        
        CLS loss - A regularization loss value is computed to prevent the CLS tokens drastically changing as they are 
        passed from block to block. As we pass the CLS tokens and BOLD tokens through each BolT block, we continually 
        embed more and more contextual information (past, present, and future) into the CLS and BOLD tokens. 
        The regularization loss value is added to the cross entropy loss value which culminates the loss value function 
        for the entire model
        
        Cross entropy - it measures the difference between the discovered probability distribution of a classification 
        model and the predicted values. When applied to binary classification tasks, may be referred to as log loss.
        Find the formulas here: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 

    inputs:
        yHat - predicted classification of subject(s) given their BOLD signal data 
        y - true classification of the subject(s)
        cls - context embeddings used to make prediction on subject(s)

    outputs:
        outputs a loss value for a set of subject(s) using the current BolT model 


    """

    def getLoss(self, yHat, y, cls):
        
        # cls.shape = (batchSize, #windows, featureDim)

        clsLoss = torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True)))

        cross_entropy_loss = self.criterion(yHat, y)  # from the __init()__ function above

        return cross_entropy_loss + clsLoss * self.hyperParams.lambdaCons


