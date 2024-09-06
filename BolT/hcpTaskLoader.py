import torch
import numpy as np

datadir = "./Dataset/Data"

"""
function:
    healthCheckOnRoiSignal(roiSignal) looks at the embedding BOLD signal embeddings and identifies any subjects with 
    embeddings that sum to 0. If a subject does have a ROI BOLD with no value then the function returns True

inputs:
    roiSignal - tensor (N, T) with ROI BOLD tokens for T observed time points

outputs:
    returns True/False depending if a ROI BOLD embedding has 0 value (returns True)
"""


def healthCheckOnRoiSignal(roiSignal):
    """
        roiSignal : (N, T)
    """


    # remove subjects with dead rois
    if(np.sum(np.sum(np.abs(roiSignal), axis=1) == 0) > 0):
        return False

    return True    


"""
function:
    abide1Loader(atlas, targetTask) loads subjects brain image data and parcels it based on a inputted brain atlas.
    This data will be transformed to images into ROI BOLD signal values. This is performed over a batch of subjects

inputs:
    atlas - brain atlas used to parcel brain images for a given subject
    targetTask - determines which dataset a given subject comes from. Takes on the value of "disease" or "control"

outputs:
    x - a collection ROI time series for a batch of subjects
    y - label of each subject (disease (0) or no disease (1))
    subjectIds - subject IDs whose information is used
"""


def abide1Loader(atlas, targetTask):

    """
        x : (#subjects, N)
    """

    dataset = torch.load(datadir + "/dataset_hcpTask_{}.save".format(atlas))

    x = []
    y = []
    subjectIds = []

    for data in dataset:
        
        if(targetTask == "gender"):
            label = int(data["pheno"]["task"])

        if(healthCheckOnRoiSignal(data["roiTimeseries"].T)):

            x.append(data["roiTimeseries"].T)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))

    return x, y, subjectIds
