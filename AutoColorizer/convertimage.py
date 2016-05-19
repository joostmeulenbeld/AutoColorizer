
from PIL import Image
import os
import numpy as np


def convert_image_to_YCbCr_and_save(jpgfilename, numpyfilename, imagesize=150):
    """Open a .jpg file, convert it to YCbCr, and save it in numpy format under numpyfilename
    INPUT:
        jpgfilename: path to the jpg file to open
        numpyfilename: path to the location to store the numpy array
        imagesize: size to scale the image to
    """
    np.save(numpyfilename, convert_image_to_YCbCr(jpgfilename, imagesize))


def convert_image_to_YCbCr(jpgfilename, imagesize=150):
    """Open a .jpg file and convert it to YCbCr
    INPUTS:
        jpgiflename: path to JPG file
        outputsize: size of the (square) image in pixels
        imagesize: size to scale the image to
    OUTPUT:
        numpy array containing the image. output.shape=(im_height, im_width, channels=3)
    """
    im = Image.open(jpgfilename)
    if im.size != (imagesize, imagesize):
        im.thumbnail((imagesize, imagesize))
    im = im.convert("YCbCr")
    # Return the layers of the image in numpy format
    return np.asarray(im)

def create_batch_and_save(batch_jpgfilenames, numpyfilename, imagesize=150):
    """given a list of jpg filenames, create a batch numpy array in correct format and save to numpyfilename
    INPUT:
        batch_jpgfilenames: list containing the paths to jpg files to be put in the batch
        numpyfilename: the path where to save the numpy file of the batch
        imagesize: size to scale the image to
    """
    nparrays = []
    for jpgfilename in batch_jpgfilenames:
        #append the image to the array with right dimensions: [channel, height, width]
        new_array = np.transpose(convert_image_to_YCbCr(jpgfilename, imagesize), [2,0,1])
        #print(jpgfilename + ", size: " + str(new_array.shape))
        nparrays.append(new_array)

    stacked_array = np.stack(nparrays)
    np.save(numpyfilename, stacked_array)


def show_YCbCr_image(numpyfilename):
    """Test function to load a numpy YCbCr array and display it (correctly using RGB colors)
    INPUT:
        numpyfilename: path to the numpy array where the image is stored in YCbCr color scale
    """
    # Load the numpy file; append .npy if it is not part of the filename
    if ".npy" not in numpyfilename:
        numpyfilename = ''.join(numpyfilename, '.npy')
    imnp = np.load(numpyfilename)
    # Create image object from numpy array - YCbCr format!
    im = Image.fromarray(imnp, mode='YCbCr')
    im.show()

def check_image_dimensions(jpgfilenames, clean=False, correctsize=(150,150,3)):
    for jpgfilename in jpgfilenames:
        shape = np.asarray(Image.open(jpgfilename)).shape

        if not (len(shape)==3 and all([correctsize[i]==shape[i] for i in range(len(correctsize))])):
            print("Incorrect: " + str(shape) + ", " + jpgfilename)
            if clean:
                os.remove(jpgfilename)

if __name__ == "__main__":
    print(convert_image_to_YCbCr_and_save("test.jpg", "test.npy", 40))
    show_YCbCr_image("test.npy")