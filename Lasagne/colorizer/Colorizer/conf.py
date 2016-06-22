# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:06:12 2016

@author: joostdorscheidt
"""

##### SETTINGS: #####
# Number of epochs to train the network over
n_epoch = 10

# Folder where the training superbatches are stored
training_folder= 'minisuperbatch'
# Folder where the validation superbatches are stored
validation_folder= 'minisuperbatch'

# The colorspace to run the NN in
colorspace= 'CIELab'

classification=False

# Parameter folder where the parameter files are stored
param_folder = 'params'
# Parameter file to initialize the network with (do not add .npy), None for no file
param_file = None
# Parameter file to save the trained parameters to every epoch (do not add .npy), None for no file
param_save_file = 'params'

# error folder where the error files are stored
error_folder = 'errors'
# Error file to append with the new training and validation errors (do not add .npy), None dont save
error_file = 'errors_fruit_YCbCr_NN_more_end_fmaps'

# The architecture to use, can be 'VGG16' or 'NN' or 'zhangNN'
architecture='zhangNN'