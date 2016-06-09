"""
The main file to run, train and evaluate the Neural network

author: Dawud Hage, written for the NN course IN4015 of the TUDelft

"""

from NNPreprocessor import NNPreprocessor
import NNVisualizer as NNshow
from NNVisualizer import gen_menu
from time import time, sleep
import numpy as np
from Colorizer import Colorizer
from glob import glob
import os
import sys

##### SETTINGS: #####
# Number of epochs to train the network over
n_epoch = 0

# Folder where the training superbatches are stored
training_folder='combination_training'
# Folder where the validation superbatches are stored
validation_folder='combination_validation' #fruit_validation'

# The colorspace to run the NN in
colorspace='CIEL*a*b*'

# Parameter folder where the parameter files are stored
param_folder = 'params'
# Parameter file to initialize the network with (do not add .npy), None for no file
param_file = None
# Parameter file to save the trained parameters to every epoch (do not add .npy), None for no file
param_save_file = 'params_combination_CIELab_mapped'

# error folder where the error files are stored
error_folder = 'errors'
# Error file to append with the new training and validation errors (do not add .npy), None dont save
error_file = 'error_combination_CIELab_mapped'

vgg16=True


######################

##### Functions: #####





######################

##### Main #####

# Load data
train_data = NNPreprocessor(batch_size=10, folder=training_folder, colorspace=colorspace, random_superbatches=True, blur=True, randomize=True)
validation_data = NNPreprocessor(batch_size=10, folder=validation_folder, colorspace=colorspace, random_superbatches=False, blur=False, randomize=False)

# Create network object
if not(param_file is None):
    param_file_loc=os.path.join(param_folder,param_file + ".npy")
else:
    param_file_loc=None
NNColorizer = Colorizer(colorspace=colorspace,param_file=param_file_loc, vgg16=vgg16)

# keep track of time
start_time_training = time()

# Keep track of the training error
train_error = 0

# Load the error file 
# Check if the training errors need to be stored in a file
last_epoch = 0 # Keep track of epoch numbers
start_epoch = 0
if not(error_file is None):
    # Check if there is a file already with the training errors
    if glob(os.path.join(error_folder,error_file) + '*'):
        # Yes it exists so open it!
        error_log = np.load(os.path.join(error_folder,error_file) + '.npy')
        last_epoch = error_log[-1,0] # last epoch stored in the error_log
        start_epoch = last_epoch
    else:
        # No so create it.
        error_log = np.empty((0,3))

else:
    # Now just put the self.error log to none
    error_log = None

# Now train the network
print("---------------------------------------")
while n_epoch > 0:
    
    

    # Train one batch
    _, error = NNColorizer.train_NN(train_data.get_batch)
    train_error += error # Add to error

    NNshow.print_progress("Progress of the training", time() - start_time_training, train_data.get_epochProgress)


    if train_data.get_epoch_done:

        # New line in the console
        print("")

        # Save the parameters!
        if not(param_save_file is None):
            NNColorizer.save_parameters(os.path.join(param_folder,param_save_file + '.npy'))

        # Keep track of the validation error
        validation_error = 0

        # reset time 
        start_time_validation = time()

        # Determine the validation error
        while not(validation_data.get_epoch_done):
            # validate the network
            _, error = NNColorizer.validate_NN(validation_data.get_batch)
            validation_error += error

            NNshow.print_progress("Progress of the validation", time() - start_time_validation, validation_data.get_epochProgress)

        # New line in the console
        print("")

        last_epoch += 1
        # store error in the error_log
        if not(error_log is None):
                error_log = np.append(error_log, np.array([[last_epoch, train_error / train_data.get_n_batches , validation_error / validation_data.get_n_batches  ]]), axis=0)
                # save the error_log
                np.save(os.path.join(error_folder,error_file + '.npy'),error_log)
                print("Stored the error values to the file: {}".format(error_file + '.npy'))

        # print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(last_epoch, start_epoch + n_epoch, time() - start_time_training))
        print("The average train error is: {!s:}".format(train_error / train_data.get_n_batches ))
        print("The average validation error: {!s:}".format(validation_error / validation_data.get_n_batches ))
        print("---------------------------------------")

        # Reset train error
        train_error = 0

        # reset time 
        start_time_training = time()
        

        if train_data.get_epoch + 1 is n_epoch:
            # Done! 
            break

##### Visualization ######
# Done with training! Lets show some images



# Now do untill the program closes:
while True:
    
    menu_options = ['Plot the erros', 'Evaluate random validation images', 'Exit the application']
    choice = gen_menu(menu_options)
    if choice == 0:
        # Plot the errors
        if not(error_log is None):
            NNshow.plot_errors(error_log)
        else:
            print("No error file provided...")
            sleep(3) # Sleep for 3 seconds to show the error, then reprint the menu
    elif choice == 1:

        print('How many images to show?')
        n_images = NNshow.get_int(range(25))

        try:
            # get random images from the validation set
            images = validation_data.get_random_images(n_images,colorspace=colorspace)

            # Run through the NN (validate to keep shape the same)
            NN_images, _ = NNColorizer.validate_NN(images)
            # Append with Luminocity layer
            if not(colorspace == 'HSV'):
                NN_images = np.append(images[:,0,:,:].reshape(images.shape[0],1,images.shape[2],images.shape[3]),NN_images,axis=1)
            else:
                NN_images = np.append(NN_images,images[:,2,:,:].reshape(images.shape[0],1,images.shape[2],images.shape[3]),axis=1)
            print(NN_images.shape)
            ## Show them :)
            NNshow.show_images_with_ab_channels(images,NN_images,colorspace)

        except:
            print("Something went wrong...")
    elif choice == 2:
        print("Are you sure?")
        menu_options = ['Yes', 'No']
        sure = gen_menu(menu_options)
        if sure == 0:
            sys.exit()
        
        

        



#from NNPreprocessor import NNPreprocessor
#from PIL import Image
#import numpy as np

#validation_folder='landscape_validation'
#validation_data = NNPreprocessor(batch_size=1, folder=validation_folder, colorspace='YCbCr', random_superbatches=False, blur=False, randomize=False)
#image = validation_data.get_random_image('YCbCr')
#image = image.reshape(3,image.shape[2],image.shape[3])
#image = np.transpose(image,(1,2,0))
## convert to PIL image
#imagep = Image.fromarray(np.uint8(image*255),'YCbCr')
#imagep.show()
    
