#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
#
from Layers import Dense
from Variables import Neurons
from Models import SequenceModel
from activations import Sigmoid
from Variables import errorFunctions

inputLayer = Dense.Dense(2,2,activation=Sigmoid,name="input layer")
outputLayer = Dense.Dense(2,1,activation=Sigmoid,name="output layer")
inputLayer.printLayerData()
outputLayer.printLayerData()

model = SequenceModel.SequenceModel(input_value=[[1,1],[1,0],[0,1],[0,0]]
                                    ,output_value=[0,1,1,0]
                                    ,error_function=errorFunctions.squard_error)
model.addLayer(inputLayer)
model.addLayer(outputLayer)
print(model.printModelData())

print("propagate:",model.propagate())
print("error:",model.computeError())
print("backpropagate:",model.backpropagate())
