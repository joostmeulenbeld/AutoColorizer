import theano.tensor as T
import theano
import numpy as np



#x = T.matrix('x')
#y = T.matrix('y')

#z = ( T.sgn(x-0.5)*( 2**(abs(x-0.5)) - 1 ) - T.sgn(y-0.5)*( 2**(abs(y-0.5)) - 1 )  )**2

#f = theano.function([x,y],[z])

#a = np.array([    [0.32 , 0.40 , 0.05 , 0.95],
#                  [0.00 , 0.50 , 0.45 , 0.48],
#                  [0.74 , 0.18 , 0.21 , 0.87],
#                  [0.27 , 0.42 , 0.75 , 0.36] ], dtype='float32')


#b = np.array([    [0.99 , 0.35 , 0.10 , 0.90],
#                  [1.00 , 0.55 , 0.50 , 0.52],
#                  [0.74 , 0.18 , 0.21 , 0.87],
#                  [0.27 , 0.42 , 0.75 , 0.36] ], dtype='float32')

#aa = np.array([    [0.00] ], dtype='float32')
#bb = np.array([    [1.00] ], dtype='float32')

#print(f(a,b))

#print(f(aa,bb))


########################################3333
""" HSV loss function test """

x = T.tensor3('x')
y = T.tensor3('y')

# Only on the first layer (here simulated as the H layer) compute the distance.
# The coordinates are circular so 0 == 1 
Hx = x[0,:,:]
Hy = y[0,:,:]

# The minimum distance on a circle can be one of three things:
# First if both points closest to eachother rotating from 0/1 CCW on a unit circle
# Second if point Hx is closer to 0/1 CCW, and point Hy CW
# Third if point Hy is closer to 0/1 CCW, and point Hx CW
Hdist = ( T.minimum( abs(Hx - Hy), 1 - T.maximum(Hx,Hy) + T.minimum(Hx,Hy)) )**2

# On the saturation layer penalize large saturation error! 
# the 2 can be changes if not saturated enough
Sx = x[1,:,:]
Sy = y[1,:,:]
Sdist = ( 2**(Sx) -  2**(Sy) )**2

# summaraze to define the loss
loss = T.sum(Sdist) + T.sum(Hdist)

f = theano.function([x,y],[loss,Hdist,Sdist])

a = np.array([[   [0.32 , 0.40 , 0.05 , 0.95],
                  [0.00 , 0.50 , 0.45 , 0.48],
                  [0.74 , 0.18 , 0.21 , 0.87],
                  [0.27 , 0.42 , 0.75 , 0.36] ],

              [   [0.32 , 0.40 , 0.05 , 0.95],
                  [0.00 , 0.50 , 0.45 , 0.48],
                  [0.74 , 0.18 , 0.21 , 0.87],
                  [0.27 , 0.42 , 0.75 , 0.36] ] ], dtype='float32')


b = np.array([[   [0.99 , 0.35 , 0.10 , 0.90],
                  [1.00 , 0.55 , 0.50 , 0.52],
                  [0.74 , 0.18 , 0.21 , 0.87],
                  [0.27 , 0.42 , 0.75 , 0.36] ],

              [   [0.99 , 0.35 , 0.10 , 0.90],
                  [1.00 , 0.55 , 0.50 , 0.52],
                  [0.74 , 0.18 , 0.21 , 0.87],
                  [0.27 , 0.42 , 0.75 , 0.36] ] ], dtype='float32')



print(f(a,b))