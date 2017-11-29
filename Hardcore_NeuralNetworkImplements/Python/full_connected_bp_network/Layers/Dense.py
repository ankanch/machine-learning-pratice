#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
# 
import numpy as np
from Variables.Neurons import Neuron
from Layers import gvLayerGlobe as GV

class Dense:
    """
    """
    name = "DenseLayer "
    neurons = []
    activation_function = None

    def __init__(self,input_size=1,output_size=1,input_values=[],ni=[],weight=1,activation=lambda x:x,name=""):
        self.neurons = ni
        GV.LAYER_COUNT+=1
        GV.LAYER_COUNT_DENSE+=1
        self.name += str(GV.LAYER_COUNT_DENSE)
        self.activation_function = activation
        if len(ni) == 0:
            if len(input_values) == 0:
                self.neurons = [ Neuron(w=[weight]*output_size) for i in range(input_size) ]
            else:
                self.neurons = [ Neuron(v=input_values[i],w=[weight]*output_size) for i in range(input_size) ]
        if len(name) != 0:
            self.name = name
            

    def activate(self,input_values):
        """
        """
        self.neurons[0].setValue(input_values[0])
        rl = self.neurons[0].computeOnWeights()
        for i in range(1,len(self.neurons)):
            self.neurons[i].setValue(input_values[i])
            rl = np.add(rl,self.neurons[i].computeOnWeights())
        return list(map(self.activation_function,rl))

    def printLayerData(self):
        print("Layer Data For [",self.name,"]")
        for i,n in enumerate(self.neurons):
            print("\t>Neuron",i,":",end='')
            n.printNeuronData()


if __name__ == "__main__":
    d1 = Dense(input_size=5,output_size=10,name="TestDense")
    d1.printLayerData()
    print(d1.activate([1,1,1,1,1]))
    d2 = Dense(input_size=3,output_size=6,input_values=[8,0,2],name="TestDense with set neuron values")
    d2.printLayerData()
    print(d2.activate([1,1,1,1,1]))
    