#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
#
from Layers import Dense
from Variables import Neurons
from Models import SequenceModel
from activations import Sigmoid

inputLayer = Dense.Dense(2,2,activation=Sigmoid,name="input layer")
outputLayer = Dense.Dense(2,1,activation=Sigmoid,name="output layer")
inputLayer.printLayerData()
outputLayer.printLayerData()

model = SequenceModel.SequenceModel(input_value=[1,1])
model.addLayer(inputLayer)
model.addLayer(outputLayer)
print(model.propagate())