#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
#
import numpy as np

class Neuron:
    """
    Stands for a single neuron in a layer.
    """

    value = np.float32(0.0)         # stores the input of current neuron
    weights = np.asarray([])     # stores weights for neurons in the next layer

    def __init__(self,v=1,w=[0,]):
        self.value = v
        self.weights = w

    def __add__(a,b):
        return a.computeOnWeights() + b.computeOnWeights()

    def setValue(self,v):
        """
        Set the neuron value: the input to the neuron.
        """
        self.value = v

    def computeOnWeights(self):
        """
        Compute the weighted input, weight*Value.

        Return a list which stands for the input of current neuron to the corresponding neurons in the next layer.
        """
        return np.multiply(self.value , self.weights)


    def printNeuronData(self):
        print(" > neuron value=",self.value,"\tWeights=",self.weights)


if __name__ == "__main__":
    a = Neuron(7,np.asarray([1,2,3,4]))
    print(a.computeOnWeights())
    b = Neuron(8,np.asarray([1,2,3,4]))
    print(b.computeOnWeights())
    print(a+b)