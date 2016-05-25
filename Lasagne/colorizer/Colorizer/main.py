## Just run the script here that you want :)

import train_network
#import show_images



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

