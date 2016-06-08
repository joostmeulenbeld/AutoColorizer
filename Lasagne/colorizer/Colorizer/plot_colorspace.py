"""
This script is meant to visualize the different colorspace conversions from RGB.
Providing a better understanding of the different colorspaces

author: Dawud Hage, written for the NN course IN4015 of the TUDelft

"""
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import pprint
pp = pprint.PrettyPrinter(indent=4, width=200)

# The idea is to use the same color conversion functions as used in the NN
# For this the conversion function needs to be imported
from NNPreprocessor import NNPreprocessor as NNPre

color_conversion = getattr(NNPre,'_convert_colorspace')

def plot_colorspace(color_array, colors, xlabel, ylabel, zlabel, axis=(0,1,2)):
    """ This sfunction plots the color array in the colors specified by colors
    INPUT:
            color_array: the coordinates to plot, shape=(3,N)
            colors: An array of colors to plot in, should be rgb, range [0, 1], shape=(N,3)
    """

    # Plot the RGB space (each point in their desired color)
    fig = plot.figure()
    ax = fig.gca(projection='3d', axisbg='black')
    ax.set_aspect('equal')

    ax.scatter(color_array[axis[0],:],color_array[axis[1],:],color_array[axis[2],:],c=colors,s=marker_size)
    ax.tick_params(colors='white')

    # Set the pane color to black
    ax.w_xaxis.set_pane_color((0,0,0))
    ax.w_yaxis.set_pane_color((0,0,0))
    ax.w_zaxis.set_pane_color((0,0,0))

    # Make the axis labels
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.set_zlabel(zlabel, color='white')
    plot.show()

 




# The function accepts an input in RGB with shape=(input_x,input_y,3)
# create a "figure", actually an array containing all possible 8bit RGB colors
# Maybe not all.. skip a few
marker_size = 100
RGB_range = np.arange(0,255,25)
R,G,B = np.meshgrid(RGB_range,RGB_range,RGB_range)

# Flatten the arrays and stack them
RGB = np.stack([R.flatten(), G.flatten(), B.flatten()]) /255.

# Reorder axis (and save for color mapping)
RGB_colormap = np.transpose(RGB,(1,0))

# Plot the RGB space
print("RGB")
plot_colorspace(RGB,RGB_colormap,'R','G','B')

# Add an axis
RGB = np.expand_dims(RGB_colormap, axis=0)

print("CIELab")
CIELab = color_conversion(image=RGB,colorspace='CIELab')
# Get rid of the extra dimensions
CIELab = np.squeeze(CIELab)

CIELab = np.transpose(CIELab, (1,0))
plot_colorspace(CIELab,RGB_colormap,'a','b','L',(1,2,0))


print("CIEL*a*b*")
CIELab = color_conversion(image=RGB,colorspace="CIEL*a*b*")
# Get rid of the extra dimensions
CIELab = np.squeeze(CIELab)
CIELab = np.transpose(CIELab, (1,0))
plot_colorspace(CIELab,RGB_colormap,'a*','b*','L*',(1,2,0))

print("YCbCr")
YCbCr = color_conversion(image=RGB,colorspace='YCbCr')
# Get rid of the extra dimensions
YCbCr = np.squeeze(YCbCr)
YCbCr = np.transpose(YCbCr, (1,0))
plot_colorspace(YCbCr,RGB_colormap,'Cb','Cr','Y',(1,2,0))

print("HSV")
HSV = color_conversion(image=RGB,colorspace='HSV')
# Get rid of the extra dimensions
HSV = np.squeeze(HSV)
HSV = np.transpose(HSV, (1,0))
plot_colorspace(HSV,RGB_colormap,'V','S','H',(1,2,0))



