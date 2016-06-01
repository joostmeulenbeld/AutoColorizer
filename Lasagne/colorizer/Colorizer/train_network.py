from Colorizer import colorizer

# Load network + add param
param_file = None
error_filename = 'error_landscape' # DO NOT add .npy!!
NNcolorizer = colorizer(param_file=param_file,error_filename=error_filename)

# Train the network, save param to file
param_save_file = 'params_landscape.npy'
NNcolorizer.train_network(3,'all','all',param_save_file)

# Show some random images
NNcolorizer.show_random_images(5)
NNcolorizer.show_random_images(5)
NNcolorizer.show_random_images(5)
# Now with UV layers
NNcolorizer.show_random_images_with_UV_channels(3)
NNcolorizer.show_random_images_with_UV_channels(3)
NNcolorizer.show_random_images_with_UV_channels(3)

