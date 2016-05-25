from Colorizer import colorizer

# Load network + add param
param_file = 'params_landscape_new_net_blur5.npy'
error_filename = 'error_landscape_new_net_blur'
NNcolorizer = colorizer(param_file=param_file,error_filename=error_filename)

# Train the network, save param to file
param_save_file = 'params_landscape_new_net_blur.npy'
NNcolorizer.train_network(10,100,2,param_save_file)

# Show some random images
NNcolorizer.show_random_images(5)
NNcolorizer.show_random_images(5)
NNcolorizer.show_random_images(5)
# Now with UV layers
NNcolorizer.show_random_images_with_UV_channels(3)
NNcolorizer.show_random_images_with_UV_channels(3)
NNcolorizer.show_random_images_with_UV_channels(3)

