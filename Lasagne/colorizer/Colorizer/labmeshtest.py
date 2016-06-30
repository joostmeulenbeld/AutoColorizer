﻿"""
This script is meant to discretize the lab colorspace. This is needed for classifying the colors into 
discrete color bins.
author: Joost Dorscheidt, written for the NN course IN4015 of the TUDelft

"""
import os
from time import time

import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plot
import matplotlib.path as path
from scipy.spatial import ConvexHull
from skimage import color


class Colorbins(object):
    
    def __init__(self, sigma = 1, k=8, T=0.2, grid_size=10, nbins=20 , y_pixels=128, labda=0.5):
        
        self.grid_size = grid_size
        # WARNING: nbins is NOT the amount of bins that will be created! It is rather a measure of the distance
        # between the bins in the ab color space
        self.nbins = nbins
        #k-nearest neihgbour parameters
        self.k = k
        self.sigma = sigma
        #annealed mean temperature
        self.T = T
        self.contour=self.get_contour()
        self.finalmesh=self.get_meshgrid()
        # Amount of bins in the classification
        self.numbins=np.shape(self.finalmesh[:,0])[0]
        self.targetvector=np.zeros([y_pixels,self.numbins])
        self.distanceindexed=np.zeros([y_pixels,self.k])
        # Current best guess of the histogram
        self._histogram = np.zeros([1,self.numbins])
        # Current amount of pixel rows used for the best guess histogram self._histogram
        self._histogramcounter = 0
        # Uniform distribution factor
        self._labda = labda
        assert self._labda != 0, print("Colorbins: labda can't be equal to 0! (otherwise you get division by zero)")
        assert self._labda < 1, print("Colorbins: labda can't be bigger than 1!")
        
    def gethistogram(self):
        return (1.-self._labda)*self._histogram + self._labda/self.numbins
        
    
    def assert_colorspace(self,colorspace):
        """ Raise an error if the colorspace is not an allowed one 
        INPUT:
                colorspace: the colorspace that the images will be put in;
                            'CIELab' for CIELab colorspace
                            'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                            'RGB' for rgb mapped between [0 and 1]
                            'YCbCr' for YCbCr
                            'HSV' for HSV
        """
        # Check if colorspace is properly defined
        assert ( (colorspace == 'CIELab') or (colorspace == 'CIEL*a*b*') or (colorspace == 'RGB') or (colorspace == 'YCbCr') or (colorspace == 'HSV') ), \
        "the colorspace must be 'CIELab' or 'CIEL*a*b*' or 'RGB' or YCbCr or 'HSV'"
    
    def _convert_colorspace(self,image,colorspace,blur=False, sigma=3):
        """ 
        INPUT:
                image: The image to be converted to the specified colorspace, should have shape=(image_x,image_y,3)
                        the input colorspace should be RGB mapped between [0 and 1], (it will return the same image if colorspace is set to RGB)
                colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YCbCr' for YCbCr
                        'HSV' for HSV
                blur: Blur the target output of the image if True.
                        Supported colorspaces:
                            'CIELab'
                            'CIEL*a*b*'
                            'HSV'
                            'YUV'
        OUTPUT:
                The image converted to the specified colorspace of shape=(image_x,image_y,3)
        """

        # Convert to CIELab
        if ( (colorspace == 'CIELab') or (colorspace == 'CIEL*a*b*') ):
            # This converts the rgb to XYZ where:
            # X is in [0, 95.05]
            # Y is in [0, 100]
            # Z is in [0, 108.9]
            # Then from XYZ to CIELab where: (DOES DEPEND ON THE WHITEPOINT!, here for default)
            # L is in [0, 100]
            # a is in [-431.034,  431.034] --> [-500*(1-16/116), 500*(1-16/116)]
            # b is in [-172.41379, 172.41379] --> [-200*(1-16/116), 200*(1-16/116)]
            image = color.rgb2lab(image)

        return image
        
    def save_histogram(self):
        """Output the histogram along with a and b locations to a numpy file and save it in folder histogram"""
        if not os.path.exists("histogram"):
            os.makedirs("histogram")
            
        savename = os.path.join("histogram", "histogram_numbins_{}".format(self.numbins))
        histogramsavename = "{}_{}".format(savename, "histogram")
        meshsavename = "{}_{}".format(savename, "mesh")
        np.save(histogramsavename, self._histogram)
        np.save(meshsavename, self.finalmesh)
        
        
    def get_colorpoints(self):
        """ 
        This script is meant to discretize the lab colorspace. This is needed for classifying the colors into 
        discrete color bins.
        INPUT:
                grid_size: the grid size of the RGB distribution
        OUTPUT:
                grid_points: all the points lying in the colorspace
        """
        #first we generate a meshgrid of R G and B values
        R_range = np.arange(0,255,self.grid_size)
        G_range = np.arange(0,255,self.grid_size)
        B_range = np.arange(0,255,self.grid_size)
        R,G,B = np.meshgrid(R_range,G_range,B_range)
        
        # Flatten the arrays and stack them (divide by 255 for the color_conversion)
        RGB = np.stack([R.flatten(), G.flatten(), B.flatten()])/255
        
        # Reorder axis (and save for color mapping)
        RGB_colormap = np.transpose(RGB,(1,0))
        
        # Add an axis (used for working with the color_conversion function of the nn_preprocessor class)
        RGB = np.expand_dims(RGB_colormap, axis=0)
        
        CIELab = self._convert_colorspace(image=RGB,colorspace='CIELab')
        # Get rid of the extra dimensions
        CIELab = np.squeeze(CIELab)
        CIELab = np.transpose(CIELab, (1,0))
        
        # project all the points on a 2d axis system (simply remove 3rd row of the array)
        CIELab = CIELab[1:,:]
        
        return CIELab
        
    def get_contour(self):
        """ This function gets the contours of the colorpoints
                
        """
        
        color_points = np.transpose(self.get_colorpoints(), (1,0))
        hull = ConvexHull(color_points)
        return color_points[hull.vertices,:]
        
    def get_meshgrid(self):
        """ This function gets the final meshgrid of the colorspace
                
        """
        color_contour=self.get_contour()
        
        #create the mesh bounded by max and min values of a and b
        a_lowbound = min(color_contour[:,0])
        a_upbound = max(color_contour[:,0])
        b_lowbound = min(color_contour[:,1])
        b_upbound = max(color_contour[:,1])
        astep = (a_upbound-a_lowbound)/(self.nbins)
        bstep = (b_upbound-b_lowbound)/(self.nbins)
        a_range = np.arange(a_lowbound,a_upbound,astep)
        b_range = np.arange(b_lowbound,b_upbound,bstep)
        amesh, bmesh = np.meshgrid(a_range,b_range)
        
        #convert mesh to a set of points
        abmesh = np.stack([amesh.flatten(),bmesh.flatten()])
        abmesh = np.transpose(abmesh,(1,0))
        
        #create a path from colorspace vertices
        color_path = path.Path(color_contour)
        points_inside = color_path.contains_points(abmesh)
        finalmesh = abmesh[points_inside]
        
        return finalmesh
    
    def k_means(self,ypixels, update_hist=True):
        """ This function gets the k-nearest neighbours of an input pixel to the colorspace meshgrid
            the distribution is smoothend with a gaussian
            INPUT:
                ypixels: shape[nrows, 2(a,b)]
                udpate_hist: update the estimated histogram or not
            OUTPUT:
                targetvector transpose: shape[nclasses, ypixels]
        """
    
        #dist=np.linalg.norm(pixel-self.finalmesh,axis=1)
        #ypixels is the input, the shape is [numrows, 2]
        #replicate matri
        #ypixels = np.tile(np.expand_dims(ypixels,axis=1), (1,self.numbins,1))
        #newmesh = np.tile(self.finalmesh, (ypixels.shape[0],1,1))
        
        if ypixels.shape[0] != self.targetvector.shape[0]:
            self.targetvector=np.zeros([ypixels.shape[0],self.numbins])
            self.distanceindexed=np.zeros([ypixels.shape[0],self.k])
        
        finalmatrix=np.tile(np.expand_dims(ypixels,axis=1), (1,self.numbins,1))-np.tile(self.finalmesh, (ypixels.shape[0],1,1))
        
        #calculate the distances
        dist = np.sum((finalmatrix)**2,axis=2)      
        
        #get the indices of the k smallest distances
        targetindices = np.argsort(dist)[:,0:self.k]

        for i in range(targetindices.shape[0]):
            self.distanceindexed[i,:] = np.array(dist[i,[targetindices[i,:]]])
            
        #apply gaussian filter over the distances
        target = np.exp(-np.power(np.sqrt(self.distanceindexed), 2.) / (2 * np.power(self.sigma, 2.)))
        self.targetvector[:,:] = 0
        
        #assign the probabilites to the targetindices
        for i in range(targetindices.shape[0]):
            self.targetvector[i,targetindices[i,:]] = target[i,:]/np.sum(target[i,:])
            
        if update_hist:
            self.update_histogram(self.targetvector)
        
        # Return the transposed vector since that format is required by the parent script
        return np.transpose(self.targetvector,(1,0))
                    
    def update_histogram(self, targetvector):
        #put the calculated distribution in the class rebalancing distribution
        self._histogramcounter += 1
        
        # Create the histogram of the input pixel column
        histogram_column = np.sum(targetvector,axis=0)
        histogram_column /= np.sum(histogram_column)
        
        # Save the new histogram using equation mean_new = (count_old*mean_old)/count_new
        histogram_new = self._histogramcounter*self._histogram + histogram_column
        self._histogram = histogram_new / (self._histogramcounter+1)                
        
    def plot_meshgrid(self, xlabel='a', ylabel='b', axis=(0,1)):
        """ This function plots the final mesh grid in the colors specified by colors
        """
    
        # Plot the RGB space (each point in their desired color)
        fig, ax= plot.subplots()
        ax.set_aspect('equal')
        
        finalmesh=np.transpose(self.finalmesh,(1,0))
        contour=np.transpose(self.contour,(1,0))
    
        ax.scatter(finalmesh[axis[0],:],finalmesh[axis[1],:],color='r')
        ax.scatter(contour[axis[0],:],contour[axis[1],:],color='g')
        ax.tick_params(colors='white')
    
        # Make the axis labels
        ax.set_xlabel(xlabel, color='white')
        ax.set_ylabel(ylabel, color='white')
        plot.show()
    
    def output_2ab(self, one_hot_vector):
        """ This function converts the target vector to ab values, where the output_vector 
        is a one-hot vector
            INPUT: one_hot_vector: a one_hot_vector where the one stands for the final ab bin.
            
            OUTPUT: ab value
        """
        return self.finalmesh[one_hot_vector,:]
        
    def annealed_mean(self, output_vector):
        """ This function applies the annealed mean to the calulated output distribution. 
        It then takes the expected value of the annealed distribution to get the final target output.
            INPUT: the output_vector: network calculated distribution among the ab bins
            
            OUTPUT: final ab value
        """
        #get the cartesian points of the output a-b values
        points = self.finalmesh[np.nonzero(output_vector),:]
        #remove one dimension
        points = np.squeeze(points)
        #get the predictions
        predictions = output_vector[np.nonzero(output_vector)]
        predictions = np.exp(np.log(predictions)/self.T)
        predictions = predictions/sum(predictions)
        
        Expected_a=0
        Expected_b=0
        
        for point, prediction in zip(points,predictions):
            #calculate the expected value 
            Expected_a=Expected_a+point[0]*prediction
            Expected_b=Expected_b+point[1]*prediction
        
        final=np.array([Expected_a,Expected_b])
            
        return final
        
        

if __name__ == "__main__":
    finalmesh=Colorbins(y_pixels=256)
    # finalmesh.plot_meshgrid()
    # t1=time()
    # y=np.zeros([10,2])
    # print(finalmesh.numbins)
    # b=finalmesh.k_means(y)
    # print(time()-t1)
    #print(b)
    #c=finalmesh.annealed_mean(b)
