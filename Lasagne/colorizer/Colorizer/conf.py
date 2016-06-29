# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:06:12 2016

@author: Dawud Hage, written for the NN course IN4015 of the TUDelft
"""

##### SETTINGS: #####
# Number of epochs to train the network over
n_epoch = 2

# Folder where the training superbatches are stored
training_folder= 'fruit_training'
# Folder where the validation superbatches are stored
validation_folder= 'fruit_validation'

# The colorspace to run the NN in
colorspace= 'CIELab'

# Parameter folder where the parameter files are stored
param_folder = 'params'
# Parameter file to initialize the network with (do not add .npy), None for no file
param_file = 'params_fruit_VGG16_concat_class_k6_T02_sigma5_nbins20_labda_08_gridsize10'
# Parameter file to save the trained parameters to every epoch (do not add .npy), None for no file
param_save_file = 'params_fruit_VGG16_concat_class_k10_T02_sigma5_nbins20_labda_05_gridsize10_normalized_value'

# error folder where the error files are stored
error_folder = 'errors'
# Error file to append with the new training and validation errors (do not add .npy), None dont save
error_file = 'errors_fruit_VGG16_concat_class_k10_T02_sigma5_nbins20_labda_05_gridsize10_normalized_value'

# The architecture to use, can be 'Dahl' or 'Compact' or 'Compact_more_end_fmaps' or 'Dahl_classifier' or 'Dahl_Zhang' or 'Dahl_Zhang_NO_VGG16'
architecture='VGG16_concat_class'

# Blur radius
sigma = 3;

# Turn on classification
classification=True

# Colorbin settings:
colorbins_k = 10 # k nearers neughbours
colorbins_T = 0.2 # Temperature
colorbins_sigma = 5 # K nearest neighbour sigma (blur the distance to the bin)
colorbins_nbins = 20 # Number of colorbins
colorbins_labda = 0.5 # Uniform distributie mix factor
colorbins_gridsize=10 
