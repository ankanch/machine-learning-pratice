#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
#
from Variables import errorFunctions 

class SequenceModel:
    """
    Stands for the sequencial model, layer after layer.
    """

    layers = []                         #stroes layers in this model
    input_value = []                    # sotres the input value list
    output_value = []                   # stores the output value list for corresponding inputs
    forward_propagation_output = []     # stores the forward propagation output 
    error_function = None               # sotres error function

    def __init__(self,layers=[],input_value=[],output_value=[],error_function=None):
        self.layers = layers
        self.input_value = input_value
        self.output_value = output_value
        self.error_function = error_function

    def addLayer(self,layer):
        """
        Add one layer to the end of current model.
        """
        self.layers.append(layer)

    def feed(self,input_values,output_values):
        """
        Feed training data.
        """
        self.input_value.extend(input_values)
        self.output_value.extend(output_values)

    def printModelData(self):
        print("Sequence Model - with",len(self.layers),"layers.")
        for i,l in enumerate(self.layers):
            print("\tLayer",i+1,end=':')
            l.printLayerData()

    def propagate(self):
        """
        Propagate from the first layer to the last layer.

        Returns the output of the last layer.
        """
        for sample in self.input_value:
            # perform forward propagation on one sample
            layer_output = sample
            for l in self.layers:
                layer_output = l.activate(layer_output)
            self.forward_propagation_output.append(layer_output)  #stores propagation output value of one sample
        return self.forward_propagation_output

    def computeError(self):
        return self.error_function(self.forward_propagation_output,self.output_value)

    def backpropagate(self):
        """
        Perform backpropagation to optimize weights.
        """
        pass
        


if __name__ == "__main__":
    pass


    