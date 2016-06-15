"""
This script is meant to discretize the lab colorspace. This is needed for classifying the colors into 
discrete color bins.
author: Joost Dorscheidt, written for the NN course IN4015 of the TUDelft

"""
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.path as path
from scipy.spatial import ConvexHull

# The idea is to use the same color conversion functions as used in the NN
# For this the conversion function needs to be imported
from NNPreprocessor import NNPreprocessor as NNPre

color_conversion = getattr(NNPre,'_convert_colorspace')

class colorbins(object):
    
    def __init__(self, k, T, grid_size=10, nbins=20):
        
        self.grid_size = grid_size
        self.nbins = nbins
        #k-nearest neihgbour parameters
        self.k = k
        self.sigma
        #annealed mean temperature
        self.T = T
        self.contour=self.get_contour()
        self.finalmesh=self.get_meshgrid()
        self.numbins=np.shape(self.finalmesh[:,0])[0]
        
        
        
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
        
        CIELab = color_conversion(image=RGB,colorspace='CIELab')
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
    
    def k_means(self,pixel):
        """ This function gets the k-nearest neighbours of an input pixel to the colorspace meshgrid
            the distribution is smoothend with a gaussian
            INPUT:
                pixel: a double where the first element is the a value and the second element is the b value
            OUTPUT:
                the vector containing the target classifications
        """
        dist = np.array([])
        for meshpoint in self.finalmesh:
            dist=np.append([dist],[np.linalg.norm(pixel-meshpoint)])
        
        targetindices = np.argsort(dist)[0:self.k]
        target = dist[targetindices]
        target = np.exp(-np.power(target, 2.) / (2 * np.power(self.sigma, 2.)))
        target = target/sum(target)
        targetvector=np.zeros([self.numbins],dtype='float32')
        targetvector[[targetindices]]=target
        return targetvector
        
            
        
        
        
        
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
        points = self.finalmesh[np.nonzero(output_vector),:]
        points = np.squeeze(points)
        predictions = output_vector[np.nonzero(output_vector)]
        print(predictions)
        predictions = np.exp(np.log(predictions)/self.T)
        print(predictions)
        predictions = predictions/sum(predictions)
        print(predictions)
        
        Expected_a=0
        Expected_b=0
        
        for point, prediction in zip(points,predictions):
            Expected_a=Expected_a+point[0]*prediction
            Expected_b=Expected_b+point[1]*prediction
        
        final=np.array([Expected_a,Expected_b])
            
        return final
        
        

#finalmesh=colorbins()
#finalmesh.plot_meshgrid()
#b=finalmesh.k_means([50,50],5,10)
#c=finalmesh.annealed_mean(b,0.1)
#print(c)



