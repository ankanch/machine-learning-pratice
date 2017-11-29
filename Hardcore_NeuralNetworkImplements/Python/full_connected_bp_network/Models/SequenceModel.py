#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
# 

class SequenceModel:
    """
    """

    layers = []
    input_value = []
    output_value = []
    forward_propagation_output = []

    def __init__(self,layers=[],input_value=[],output_value=[]):
        self.layers = layers
        self.input_value = input_value
        self.output_value = output_value

    def addLayer(self,layer):
        """
        """
        self.layers.append(layer)

    def feed(self,input_values,output_values):
        """
        """
        self.input_value.extend(input_values)
        self.output_value.extend(output_values)

    def printModelData(self):
        """
        """
        pass

    def propagate(self):
        """
        """
        self.forward_propagation_output = self.input_value
        for l in self.layers:
            self.forward_propagation_output = l.activate(self.forward_propagation_output)
        return self.forward_propagation_output

    def backpropagate(self):
        """
        """
        pass


if __name__ == "__main__":
    pass


    