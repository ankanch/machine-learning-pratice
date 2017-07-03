import numpy as np
import activiation

class ANNLayer:
    """
         This class represents a single layer of an
         artificial neural network.
         function in this class will only works for current layer.


         @`by Kanch -> kanchisme@gmail.com`@
    """

    var_neuron_amount = 2
    var_bias_value = 0
    var_layer_name = "defualt"
    var_neurons = [0,0]

    def __init__(self,neuron_amount,bias=0,name="default"):
        self.var_neuron_amount = neuron_amount
        self.var_bias_value = bias
        self.var_layer_name = name
    
    def setNeuron(self,amount,values=[]):
        """
        Set the neurons in this layer. Parameter `amount` is the number of neurons, at least 2.`values` is used to set 
        the values of there neurons, if this layer is input layer that will be useful
        """
        self.var_neuron_amount = amount
        self.var_neurons = ([0] * amount).copy()
        if len(values) == amount:
            self.var_neurons = values.copy()



    def setBias(self,value=0):
        """
        Set the bias unit value to current layer
        If `0` is assigned, it will be treated as no bias unit.
        """
        self.var_bias_value = value

    def forward_propagation(self,theta):
        """
        This function is used to perfom forward propagation with given weight for current layer.
        This will return the input to the next layer, a list.`theta` can be a  2D list
        """
        # just convert var_neurons to a column vector by using np.matrix
        inputVc = np.matrix(self.var_neurons)
        inputVc = inputVc.reshape(self.var_neuron_amount,1)
        print("after convert to array\n",inputVc)

        # convert theta to a row vector
        thetaVr = np.matrix(theta)
        print("weight vector\n",thetaVr)

        # then we can use it to take multiplication with weight array
        hiv = np.dot(thetaVr,inputVc)
        print("after make dot product:\n",hiv)

        # now apply activiation ( sigmoid ) to the reuslt vector
        hiv2 = np.apply_along_axis(activiation.sigmoid,1,hiv)
        print("after apply sigmoid function:\n",hiv2)

        return hiv2
        

    def printLayerInfo(self):
        """
        Print basic infomation of current layer, like name, amount of neurons, bias value.
        """
        print("Layer",self.var_layer_name,"have",self.var_neuron_amount,\
                "neurons with bias value of",self.var_bias_value)
        print("Value of neurons are",self.var_neurons)

    


if __name__ == "__main__":
    print("test for network:")
    la = ANNLayer(3,1,"input")
    la.setNeuron(4,[4,5,8,2])
    la.setBias(0)
    la.printLayerInfo()
    la.forward_propagation([[1,2,3,4],[1,2,3,4],[4,5,6,7],[9,1,0,2]])
    