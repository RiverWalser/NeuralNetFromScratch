from newneuralnetwork import Neuron, Network, DataModifyer


import numpy as np
datamodifyer = DataModifyer(3)
train_set = datamodifyer.create_data_set()
val_set = datamodifyer.create_data_set()


NeuralNet = Network([7,7], [7,7], [7,7], 3, 3, 5, 3, "mse", "relu", train_set, val_set, "default", "default",.01)

NeuralNet.train()  

