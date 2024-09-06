import torch
from einops import rearrange, repeat
import numpy as np

"""
function:
    windowBoldSignal(boldSignal, windowLength, stride) generates a list of matrices (or related object)
    comprising of different windows of BOLD signals. After all windows have been created they are concatenated
    together. Additionally, endpoints of the windows are collected

inputs:
    boldSignal - 3-d tensor object that contains BOLD signals from individual ROIs and various subjects
    windowLength - Length of BOLD signal window
    stride - Denotes temporal distance between the beginning time of one time window and the next

outputs:
    windowedBoldSignals - 1-d tensor object that contains (check on dimension) that contains all the BOLD signals
                          in their respective time windows
    samplingEndPoints - a list containing the indices for window endpoints

"""


def windowBoldSignal(boldSignal, windowLength, stride):
    
    """
        boldSignal : (batchSize, N, T)
        output : (batchSize, (T-windowLength) // stride, N, windowLength )
    """

    T = boldSignal.shape[2]

    # NOW WINDOWING 
    windowedBoldSignals = []
    samplingEndPoints = []

    for windowIndex in range((T - windowLength)//stride + 1):

        # Pulls all BOLD signals from a 3-d tensor object (3rd dimension is time other.
        # Other dimensions represent ROIs and subjects (double check this not 100% sure)
        sampledWindow = boldSignal[:, :, windowIndex * stride  : windowIndex * stride + windowLength]
        samplingEndPoints.append(windowIndex * stride + windowLength)

        sampledWindow = torch.unsqueeze(sampledWindow, dim=1)  # returns tensor object of given dimension
        windowedBoldSignals.append(sampledWindow)

    # concatenates the list of tensors into single tensor of given dimension(s)
    windowedBoldSignals = torch.cat(windowedBoldSignals, dim=1)

    return windowedBoldSignals, samplingEndPoints
