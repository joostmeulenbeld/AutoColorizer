import lasagne
import theano.tensor as T


class logSoftmax(lasagne.layers.Layer):
    def __init__(self,input):
        super(logSoftmax, self).__init__(input)
        

    def get_output_for(self, input, **kwargs):
        xdev = input - input.max(1, keepdims=True)
        return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
    