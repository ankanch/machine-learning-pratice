#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
#
import numpy as np
from NumpyFunctionWrapper import FunctionWrapper,TypeWrapper 

class Neuron:
    """
    """

    value = TypeWrapper.kfloat(0.0)
    weights = FunctionWrapper.klist([])

    def __init__(self,v=1,w=[0,]):
        self.value = v
        self.weights = w

    def __add__(a,b):
        """
        """
        return a.computeOnWeights() + b.computeOnWeights()

    def setValue(self,v):
        """
        """
        self.value = v

    def computeOnWeights(self):
        """
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