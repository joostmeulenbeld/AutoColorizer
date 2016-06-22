import theano.tensor as T
import lasagne


class CustomSoftmax(lasagne.layers.Layer):
    """
        INPUT:
                shape[batch size, num classes, xpixels ypixels]
        OUTPUT:
                shape[batch size*xpixels*ypixels,numclasses]
    """
    def __init__(self, incoming, num_bins, batch_size, x_pixels, y_pixels, **kwargs):
        super(CustomSoftmax, self).__init__(incoming, **kwargs)
        self.num_bins = num_bins
        self.batch_size = batch_size
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
    
    
    def get_output_for(self, input, **kwargs):
        input_dimshuffled = input.dimshuffle((0,2,3,1))
        input_reordered = input_dimshuffled.reshape((self.batch_size*self.x_pixels*self.y_pixels,self.num_bins))
        
        return T.nnet.softmax(input_reordered)
        
    def get_output_shape_for(self, input_shape):
        return (self.batch_size*self.x_pixels*self.y_pixels,self.num_bins)