from Colorizer import colorizer

# Load network + add param
param_file = 'params_20_05_16_20_new_net_batches_100_epoch_20_morning_final.npy'
NNcolorizer = colorizer(param_file=param_file)

# Train the network, save param to file
param_save_file = 'params_21_05_16__170_epoch_total.npy'
NNcolorizer.train_network(50,100,2,param_save_file)

# Show some random images
NNcolorizer.show_random_images(5)
NNcolorizer.show_random_images(5)
NNcolorizer.show_random_images(5)
# Now with UV layers
NNcolorizer.show_random_images_with_UV_channels(3)
NNcolorizer.show_random_images_with_UV_channels(3)
NNcolorizer.show_random_images_with_UV_channels(3)

