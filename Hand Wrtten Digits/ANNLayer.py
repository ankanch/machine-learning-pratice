import numpy as np
import random
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

    def __init__(self,neuron_amount,bias=0,name="default",value=[]):
        self.var_neuron_amount = neuron_amount
        self.var_bias_value = bias
        self.var_layer_name = name
        if len(value) == neuron_amount:
            self.var_neurons = value.copy()

    def getNeuronValues(self):
        return self.var_neurons.copy()
    
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
        Set the bias unit value to current layer.
        If `0` is assigned, it will be treated as no bias unit.
        """
        self.var_bias_value = value

    def forward_propagation(self,theta):
        """
        This function is used to perfom forward propagation with given weight for current layer.
        This will return the input to the next layer, a list.
        * `theta` need to be a  2D list
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

    def randomInitTheta(self,roundv=0):
        """
        This function will random initialize a theta matrix for current layer.
        This usually used in the first run of this network.
        This function will return a matrix depend on the size of this layer.
        """
        ssize = self.var_neuron_amount
        thetax = [0] * ssize
        thetax = [thetax]*ssize
        
        #then we initialize these value into [0,1] 
        i=0
        j=0
        for vector in thetax:
            for value in vector:
                if roundv == 0:
                    thetax[i][j] = random.uniform(0, 1)
                else:
                    thetax[i][j] = round(random.uniform(0, 1),roundv)
                j+=1
            i+=1
            j=0
        return thetax
        

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
    # example neural network with 1 input, 1 hidden and 1 output
    # network size: 3*3*2
    print("tset for a example neural network with 1 input layer,1 hidden and 1 output.(3x3x2)")
    iv = [10,20,5]
    la_input = ANNLayer(3,0,"input",iv)
    la_hidden = ANNLayer(3,0,"hidden 1")
    la_output = ANNLayer(2,0,"output")
    print("input list is:\n",iv)
    theta_12 = la_input.randomInitTheta(2)
    theta_23 = la_hidden.randomInitTheta(2)
    print("random initialized theta matrix from layer 1 to 2:\n",theta_12)
    print("random initialized theta matrix from layer 2 to 3:\n",theta_23)
    print("forward propagation from layer 1 to layer 2:\n")
    la_hidden.setNeuron(la_hidden.var_neuron_amount,la_input.forward_propagation(theta_12))
    la_hidden.printLayerInfo()
    print("forward propagation from layer 2 to layer 3:\n")
    la_output.setNeuron(la_output.var_neuron_amount,la_hidden.forward_propagation(theta_23))
    la_output.printLayerInfo()
    print("neural network output list is:\n")

    

    