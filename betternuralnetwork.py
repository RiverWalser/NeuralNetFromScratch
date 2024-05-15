import numpy as np

class Network:
    def __init__(self, input_dim, output_dim, hidden_layers, input_layer_dim, output_layer_dim, hidden_dim):
        #instead of having a neuron class represent an individual neuron, a single neurons data is scattered and its position the individual
        
        #Because a matrix multiplied by its transpose is a square, and a square multiplied by its own dimension is a square,
        #we are going to take in a matrix of any demension, the input layer will have weights of its transpose (demension only, the actuall numbers are random)
        #then the activations are now a square, the entire hidden layer can now be square. 
        #after the hidden layer, we need to find out the dimensions of matrix C so that when the activations from the hidden layer
        #are multiplied by C, its the same demensions as the output
        self.weights = []
        for i in range(hidden_layers + 2):#for each layer
            self.weights.append([])#create a layer
            if i == 0:#at the first layer
                print("at first layer")
                for j in range(input_layer_dim):#for the amount of neurons in that layer
                    self.weights[i].append([])#add a neuron               
                    """
                    the demensions of a perticular weight is the transpose of a weight in the previous layer
                    """
                    x = input_dim[1]
                    y = input_dim[0]
                
                #if its even, its x=1,y=0 if its odd its the transpose
                    for m in range(x):
                        self.weights[i][j].append([])#create a new layer
                        for n in range(y):
                            self.weights[i][j][m].append(np.random.randint(-100, 100))#append a weight element
                print(f'weights in first layer are: {self.weights[i]}')
            elif i == hidden_layers + 1:
                print("at last layer")
                A_dim = (input_dim[0], input_dim[0]) # The demensions of the matrix that is propogated through the hidden layers
                C_dim = (output_dim[0], output_dim[1])
                if len(A_dim) != 2 or len(C_dim) != 2:
                    raise ValueError("Input dimensions must be tuples of length 2.")
    
                if A_dim[0] != C_dim[0]:
                    raise ValueError("Number of rows in A must match number of rows in C.")
    
                B_dim = (A_dim[1], C_dim[1])


                for j in range(output_layer_dim):
                    
                    self.weights[i].append([])#add a neuron               
                    #Create a matrix that is the same demensinos as B
                    for m in range(B_dim[0]):
                        self.weights[i][j].append([])#create a new layer
                        for n in range(B_dim[1]):
                            self.weights[i][j][m].append(np.random.randint(-100, 100))#append a weight element
            else:
                print(f"Hidden Layer,layer{i}")
                for j in range(hidden_dim):
                    self.weights[i].append([])#add a neuron               
                    """
                    the demensions of a perticular weight is the transpose of a weight in the previous layer
                    """
                    l = input_dim[0]# the product of a matrix multiplied by its transpose is a square of the demensions of the first row
                    k = input_dim[0]
                    for x in range(l):
                        self.weights[i][j].append([])

                        for y in range(k):
                            self.weights[i][j][x].append(np.random.randint(-100, 100))#append a weight element
            print(self.weights)
            
            
        
        #now we need to create the activation placeholders
        self.activations = []
        for i in range(hidden_layers + 2):#for each layer
            self.activations.append([])#create a layer
            #all layers but the last layer will be a square of demensions of the input[0]
            if i == 1:
                for j in range(input_layer_dim):
                    self.activations[i].append([])
                    for x in range(input_dim[0]):#the amount of rows and collumns are both the amount of rows in the input matrix, as i have shown above
                        self.activations[i][j].append([])#create a row inside of the matrix
                        for y in range(input_dim[0]):
                            self.activations[i][j][x].append(0)#add a 0(its a placeholder)
            elif i == hidden_layers + 1:
                for j in range(output_layer_dim):
                    self.activations[i].append([])
                    for x in range(output_dim[0]):
                        self.activations[i][j].append([])#create a new row
                        for y in range(output_dim[1]):
                            self.activations[i][j][x].append(0)
            else:
                for j in range(hidden_dim):
                    self.activations[i].append([])
                    for x in range(input_dim[0]):#the amount of rows and collumns are both the amount of rows in the input matrix, as i have shown above
                        self.activations[i][j].append([])#create a row inside of the matrix
                        for y in range(input_dim[0]):
                            self.activations[i][j][x].append(0)#add a 0(its a placeholder)               

        self.biases = []
        for i in range(hidden_dim + 2):
            if i == 0:
                for j in range(input_layer_dim):
                    self.activations.append(np.random.randint(-50, 50))
            elif i == hidden_layers + 1:
                for j in range(output_layer_dim):
                    self.activations.append(np.random.randint(-50, 50))
            else:
                for j in range(hidden_dim):
                    self.activations.append(np.random.randint(-50, 50))
                    
        print(f'weights: {self.weights}')
            #print(f'activations: {self.activations}')
            #print(f'biases: {self.biases}')
                    
        
        

N = Network([3,2], [3, 4], 5, 3, 3, 6)
                
                
                
        
        