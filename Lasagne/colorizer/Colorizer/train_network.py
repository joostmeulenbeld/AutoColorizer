from Colorizer import Colorizer

# Load network + add param
param_file = None #'params_landscape.npy'
error_filename = None #'error_landscape' # DO NOT add .npy!!
NNcolorizer = Colorizer(param_file=param_file,error_filename=error_filename)

# Train the network, save param to file
param_save_file = None #'params_landscape.npy'
NNcolorizer.train_network(4,1,'all',param_save_file)

# Show some random images
NNcolorizer.show_random_images(5)
NNcolorizer.show_random_images(5)
NNcolorizer.show_random_images(5)
# Now with UV layers
NNcolorizer.show_random_images_with_UV_channels(3)
NNcolorizer.show_random_images_with_UV_channels(3)
NNcolorizer.show_random_images_with_UV_channels(3)

