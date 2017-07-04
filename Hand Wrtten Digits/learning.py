from ANNLayer import ANNLayer

def backpropagation(output,weight):
    pass

def forwardpropagation(layers,weights,inputs=[]):
    """
    This function takes a list of layers and a list of theta matrix.
    Then this function will run forwad propagation to it.Then return the output vector    
    * `inputs`: the raw input(single list) to the first layer,if not specified, assume input is already set in layer 1
    * `layers`: the layer list that contains input, hidden and output layers, at least two layers
    * `weights`: list of weight (theta) matrix
    """
    first = True
    weight_pos = 0
    layer_pos = 1
    for layer in layers:
        if first:
            if len(inputs) == layer.var_neuron_amount:
                layer.setNeuron(layer.var_neuron_amount,inputs)
            layers[layer_pos].setNeuron(layers[layer_pos].var_neuron_amount,layer.forward_propagation(weights[weight_pos]))
            first = False
        else:
            layers[layer_pos].setNeuron(layers[layer_pos].var_neuron_amount,layer.forward_propagation(weights[weight_pos]))
        weight_pos+=1
        layer_pos+=1
        if layer_pos == len(layers):
            break
    return layers[len(layers)-1].getNeuronValues()

if __name__ == "__main__":
    print("test for learning:\ncreating network")
    iv = [10,20,5]
    la_input = ANNLayer(3,0,"input",iv)
    la_hidden = ANNLayer(3,0,"hidden 1")
    la_output = ANNLayer(2,0,"output")
    las = [ la_input,la_hidden,la_output ]
    theta_12 = la_input.randomInitTheta(2)
    theta_23 = la_hidden.randomInitTheta(2)
    thetas = [ theta_12 , theta_23 ]
    print("start perform forward propagation.\n")
    result = forwardpropagation(las,thetas,iv)
    print("result vector:",result)
    las[len(las)-1].printLayerInfo()