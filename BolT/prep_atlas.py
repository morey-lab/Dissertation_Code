
import os 
import nilearn as nil
import nilearn.datasets


datadir = "./Dataset/Data"

"""
function:
    prep_atlas(atlas) takes in a given atlas argument (only operates on schaefer7_400) and parcels the image and 
    captures ROI signal information

inputs:
    atlas - names a brain atlas to determine ROI structure and values

outputs:
    atlasImage - a 3D/4D Niimg-like object with new attributes
"""

def prep_atlas(atlas):

    if(atlas == "schaefer7_400"):
        if(not os.path.exists(datadir + "/Atlasses/{}".format(atlas))):
            atlasInfo = nil.datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1, data_dir=datadir + "/Atlasses")            
            atlasImage = nil.image.load_img(atlasInfo["maps"])

    return atlasImage


    