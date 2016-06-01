## Just run the script here that you want :)

#import train_network

from NNPreprocessor import NNPreprocessor

beun = NNPreprocessor(10,'bla')

#import show_images

#import numpy as np
#import matplotlib.pyplot as plot

#errors = np.load('error_landscape.npy')
#plot.plot(errors[:,1], label='Train error')
#plot.plot(errors[:,2], label='Validation error')
#plot.legend()
#plot.show()


## Blur one image the U and V layers
#import numpy as np
#import matplotlib.pyplot as plot
#from PIL import Image
#from scipy.ndimage.filters import gaussian_filter
## Open a batch
#batch = np.load('training/batch_0.npy')
#img_id = 5
#Y = batch[img_id,0,:,:]
#U = batch[img_id,1,:,:]
#V = batch[img_id,2,:,:]

#img = np.zeros((128,128,3), 'uint8')
#img[..., 0] = Y
#img[..., 1] = U
#img[..., 2] = V
#Img = Image.fromarray(img,'YCbCr')
## blur
#sigma = 2
#U_b = gaussian_filter(U,sigma)
#V_b = gaussian_filter(V,sigma)
#img_b = np.zeros((128,128,3), 'uint8')
#img_b[..., 0] = Y
#img_b[..., 1] = U_b
#img_b[..., 2] = V_b
#Img_b = Image.fromarray(img_b,'YCbCr')
## Show image
#f, ax = plot.subplots(2,3)
#index = 0
## Show original image
#ax[index,0].axis('off')
#ax[index,0].imshow(Img)
## show the U layer
#ax[index,1].axis('off')
#ax[index,1].imshow(U,cmap='gray')
## Show the V layer
#ax[index,2].axis('off')
#ax[index,2].imshow(V,cmap='gray')

## Show the NN image
#ax[index+1,0].axis('off')
#ax[index+1,0].imshow(Img_b)
## show the U layer
#ax[index+1,1].axis('off')
#ax[index+1,1].imshow(U_b,cmap='gray')
## Show the V layer
#ax[index+1,2].axis('off')
#ax[index+1,2].imshow(V_b,cmap='gray')
#plot.show()

#import numpy as np
#import os
#import matplotlib.pyplot as plot

#plot.axis([0,1000,0,1])
#plot.ion()

#data = np.empty((0,3))

#for i in range(100):
#    plot.cla()
#    temp_y=np.random.random()*i
#    temp_z=np.random.random()
#    data = np.append(data, np.array([[i,temp_y, temp_z]]), axis=0)
#    plot.plot(data[:,0],data[:,1], label='first')
#    plot.plot(data[:,0],data[:,2], label='second')
#    plot.legend()
#    plot.draw()
#    plot.pause(0.01)

#folder = 'training'
#filename = os.listdir(folder)[0]
#batch_file_path = os.path.join(folder, filename)
#batch = np.load(batch_file_path)/np.float32(256)
#Umin = 0
#Umax = 0
#Vmin = 0
#Vmax = 0
#batches = 25
#for im in range(batches):
#    Umax += np.amax(batch[im,1,:,:])
#    Umin += np.amin(batch[im,1,:,:])
#    Vmax += np.amax(batch[im,2,:,:])
#    Vmin += np.amin(batch[im,2,:,:])

#print("the average maximum of the U layers in the batch: " + str(Umax/batches))
#print("the average minimum of the U layers in the batch: " + str(Umin/batches))
#print("the average maximum of the V layers in the batch: " + str(Vmax/batches))
#print("the average minimum of the V layers in the batch: " + str(Vmin/batches))


