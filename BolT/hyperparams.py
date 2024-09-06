
#This is not from utils package, but a Python script
from transformer_models.BolT.utils import Option

"""
function:
        getHyper_bolT() creates a dictionary that holds all hyperparameters necessary for the BolT transformer model. 
        The Option() is a class that has two functions:
                The first __init__() goes through the hyperDict object creates another dictionary which is identical to 
                hyperDict, but is now assigned to the Option class
                
                The second function, copy(), creates a deep copy of the dictionary created by the __init__() function

inputs:


outputs:
        hyperDict- a dictionary object that holds all hyperparameters necessary for the BolT transformer model. 
        Additionally Option() builds in several new functions to the dictionary object
"""


def getHyper_bolT():

    hyperDict = {

            "weightDecay" : 0,

            "lr" : 2e-4,
            "minLr" : 2e-5,
            "maxLr" : 4e-4,

            # FOR BOLT
            "nOfLayers" : 4,
            "dim" : 400,

            "numHeads" : 36,
            "headDim" : 20,

            "windowSize" : 20,
            "shiftCoeff" : 2.0/5.0,            
            "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
            "focalRule" : "expand",

            "mlpRatio" : 1.0,
            "attentionBias" : True,
            "drop" : 0.1,
            "attnDrop" : 0.1,
            "lambdaCons" : 1,

            # extra for ablation study
            "pooling" : "cls", # ["cls", "gmp"]         
                

    }

    return Option(hyperDict)

