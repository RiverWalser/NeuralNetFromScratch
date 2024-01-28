import numpy as np

class DataModifyer:
    def __init__(self,x):
        self.x = x


    def generate_integer_matrix(self):
        return np.random.randint(1, 10, size=(7, 7))
    def create_data_set(self):
        data_set = []
        for _ in range(20):
            matrix1 = self.generate_integer_matrix()
            matrix2 = matrix1 * 2
            data_set.append([matrix1, matrix2])

        for pair in data_set:
            matrix1, matrix2 = pair
            for row in range(7):
                print(' '.join(f'{elem:2d}' for elem in matrix1[row]), end=' | ')
                print(' '.join(f'{elem:2d}' for elem in matrix2[row]))
            print(' ' * 10) # Add 10 spaces between pairs
        return data_set



    
 #Add connections       
class Neuron:
    def __init__(self, layer, position, weight_shape, vec_shape, activation, connection, conn_weight):
        """_summary_

        Args:
            layer (int): shows which layer the neuron is on(1 would be the first layer)
            position (int): shows where the neuron is in its layer (used mainly to identify a specific neuron)
            weight_shape (list): the demensions of the weight matrix
            vec_shape (list): The demensions of the neurons intrinsic vector(when a vector is being propogated through the network, this tensor stores the vector)
            activation (str): shows which activation function we want
            connection (list): a list of all the neurons places in the previous layer, this neuron will only take values from neurons whose place is in that list
            conn_weight (list): weighted connections, a digit which is multiplied to the vector coming from that neuron, keep it full, if a neuron doesnt have a connection to it, then just make the weight zero or else it wont work
            
        Generated Values:
            vector(list)- a matrix with demensions specified in vec_shape
            weight(list) - the weight matrix of the neuron(initialised following the standard normal distribution)
            bias (intiger)- a random value from 1 to 100 which will be added to every element in the final vector
        """
        self.vec_shape = vec_shape
        self.weight_shape = weight_shape
        self.vector = np.zeros((vec_shape[0], vec_shape[1]))
        self.weight = np.random.standard_normal((weight_shape[0], weight_shape[1]))
        self.bias = np.random.randint(0, 100)
        self.layer = layer
        self.position = position
        self.connections = connection
        self.activation = activation
        self.conn_weight = conn_weight
        self.conn_weight = conn_weight
    
    def average_vec(self, inputvecs):
        print(self.vector)
        #Since now were dealing with tensors,(the inputvec is a 3d tensor), find the average of each position
        #Inputvec tensor should be something like                                                                                       
        """
        [T111, T112, T113, ... T11n],      |      [Tn11, Tn12, Tn13, ... Tn1n],     
        [T121, T122, T123, ... T12n],      |      [Tn21, Tn22, Tn23, ... Tn2n],              
        [T131, T132, T133, ... T13n],      |      [Tn31, Tn32, Tn33, ... Tn3n],
        [  .     .     .   .       ],      |      [  .     .     .   .       ],
        [  .     .     .     .     ],      |      [  .     .     .     .     ],
        [  .     .     .       .   ],      |      [  .     .     .       .   ],
        [T1n1, T1n2, T1n3, ... T1nn],      |      [Tnn1, Tnn2, Tnn3, ... Tnnn]
        
        each matrix in the inputvec tensor is the same demension as the vector matrix, we can use that 
        
        were going to index through a matrix, and for each position find the average 
        
        
        """

 

        average_vector = np.zeros((self.vec_shape[0], self.vec_shape[1]))

        for i in range(len(inputvecs[0])):
            for j in range(len(inputvecs[0][0])):
                num = 0
                p = 0
                for k in range(len(inputvecs)):
                    if k in self.connections:# if this neuron has a connection to the neuron in the other layer
                        num += inputvecs[k][i][j] * self.conn_weight[p] # you can weight connections to be more or less
                        p += 1
                num /= p
                average_vector[i][j] = num

        print(f'Averaged Vector: {average_vector}')
        
        return average_vector
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def grad_activation(self, x):
        print(f'X: {x}')
        print(f"layer: {self.layer}")
        if self.activation == 'relu':
            if x > 0:
                return x
            else:
                return 0
        elif self.activation == "sigmoid":
            return self.sigmoid(x)
                    
        else:
            return x
    def activation_derivative(self, x):
    # Define the derivative of your activation function here
        return self.grad_activation(x) * (1 - self.grad_activation(x))

    def propogate(self, vectors):
        
        avg_vec = self.average_vec(vectors)
        weighted_vec = np.dot(avg_vec, self.weight)
        for i in range(len(weighted_vec)):
            for j in range(len(weighted_vec[0])):
                weighted_vec[i][j] += self.bias
                weighted_vec[i][j] = self.grad_activation(weighted_vec[i][j])
                
        self.vector = weighted_vec
        
    def show(self):
        print(f'Vector: {self.vector}')
        print(f'Weight matrix:{self.weight}')
class Network:
    def __init__(self, input_matrix_deminsions, weight_matrix_demensions, output_matrix_demensions, input_layer_demensions, hidden_layers, hidden_layer_demensions, output_layer_demensions, lossfunction, activation, train_set, val_set, connection, conn_weight, learning_rate):
        #define variables
        self.input_layer = []
        self.hidden_matrix = []

        self.output_layer = []
        self.input_matrix_deminsions = input_matrix_deminsions
        self.weight_matrix_demensions = weight_matrix_demensions
        self.output_matrix_demensions = output_matrix_demensions
        self.input_layer_demension = input_layer_demensions
        self.hidden_layers = hidden_layers
        self.hidden_layer_demensions = hidden_layer_demensions
        self.output_layer_demensions = output_layer_demensions
        self.loss_function = lossfunction
        self.activation = activation
        self.train_set = train_set
        self.val_set =  val_set
        self.connections = connection
        self.learning_rate = learning_rate
        if self.connections == "default":
            self.standard_connection  = np.arange( max(input_layer_demensions, output_layer_demensions, hidden_layer_demensions)+ 1)
        self.conn_weight = conn_weight
        if self.conn_weight == "default":
            self.connection_weights = np.ones(max(input_layer_demensions, hidden_layer_demensions, output_layer_demensions))
        
        for i in range(self.input_layer_demension):
            self.input_layer.append(Neuron(1, i, self.weight_matrix_demensions, self.input_matrix_deminsions, self.activation,self.standard_connection,self.connection_weights))
        for i in range(self.hidden_layers):
            self.hidden_matrix.append([])#create a new layer for the neurons to go in
            #possible error: you need to check that the list you just created is the same one that gets the neurons appended to it
            for j in range(self.hidden_layer_demensions):
                self.hidden_matrix[i].append(Neuron(i, j, self.weight_matrix_demensions, self.input_matrix_deminsions, self.activation,self.standard_connection,self.connection_weights))
        for i in range(self.output_layer_demensions):
            self.output_layer.append(Neuron(1+self.hidden_layer_demensions, i, self.weight_matrix_demensions, self.input_matrix_deminsions,"relu",self.standard_connection,self.connection_weights))
 

        
    def forward_propogation(self, input_matrix):
        for i in range(len(self.input_layer)):
            print(f'input matrix: {input_matrix}')
            self.input_layer[i].propogate(input_matrix)
            print(f'New Vector for neuron: {i} : {self.input_layer[i].vector}')
        #find the input matrices at the begging of the row to save time complexity
        for i in range(len(self.hidden_matrix)):#for each layer
            if i == 0:#on the first layer
                inputs = []
                for k in range(len(self.input_layer)):
                    inputs.append(self.input_layer[k].vector)  
                print(f'First Layer inputs: {inputs}')
            #on the first layer set the input tensor to a list of the input layers vectors       
            else:
                    inputs = []
                    for k in range(len(self.hidden_matrix[i - 1])):#loop through previous layer
                        inputs.append(self.hidden_matrix[i-1][k].vector)#append all previous layers output vectors to a tensor
                    
            for j in range(len(self.hidden_matrix[i])):
                #create list of input layer matrices
                print(f'Layer :{i}, the inputs are: {inputs}')
                self.hidden_matrix[i][j].propogate(inputs)
        inputs = []
        
        #gather the vectors of the last layer in the hidden layer so we can propogate it to the output layer
        for i in range(len(self.hidden_matrix[-1])):
            inputs.append(self.hidden_matrix[-1][i].vector)
        for i in range(len(self.output_layer)):
            self.output_layer[i].propogate(inputs)
            print("got here")
            
        outputs = []
        for i in range(len(self.output_layer)):
            outputs.append(self.output_layer[i].vector)
        return self.elementwise_average(outputs)
       

    def elementwise_average(self, matrix_list):
        # Get the number of matrices in the list (n)
        n = len(matrix_list)
        
        # Calculate the sum of all matrices element-wise
        sum_of_matrices = np.sum(matrix_list, axis=0)
        
        # Divide the sum by n
        averaged_matrix = sum_of_matrices / n
        
        return averaged_matrix
                 
    def Loss_Function(self, predicted_value, output_vec):
        squared_difference = (predicted_value - output_vec) ** 2
    
    # Calculate the mean of the squared differences
        mse = np.mean(squared_difference)
    
        return mse
    """
    def back_propagation(self, predicted_value, output_vec,learning_rate):
        
        
        """
        #Step 1: create delta value
        #    difference of the two matrices?
            
"""
        
        delta_output = predicted_value - output_vec
        print(f'delta output: {delta_output}')
        """
        #Step 2: backpropogate gradient through outer layer
        #I think this means i need to calculate the gradient of only the outer layer
        #lets index through the output layer and calculate the gradients for one neuron at a time
        #we need to calculate that layers derrivitive of the loss with respect to the activations of the output

"""
        Loss = self.Loss_Function(predicted_value, output_vec)
        Dloss_DA = 2 * (predicted_value - output_vec)
        print(f'Derrivitave of the loss with respect to the activations: {Dloss_DA}')
        
        """
        #Dloss_DA was actually a gradient of the entire row, if there are three neurons in a row, it takes the 
        #element at [0][0] from each neuron and finds the total loss all three of them contribute to
        
        
        #now that we have that, we will go through each neurons weight matrix and change it
        
        #Steps:
        #    index through the output layer
        #   for each neuron index through its weight matrix and change each element
            
        #    we also need to calculate the gradient for each neuron
"""
        rows = len(self.output_layer[0].weight)
        cols = len(self.output_layer[0].weight[0])
        dLoss_dW_output =  np.zeros((len(self.output_layer), rows, cols))
        for i in range(len(self.output_layer)):#for each neuron
            dLoss_dW_output[i] = np.outer(Dloss_DA[i], self.output_layer[i].values)#calculate gradient for that neuron
            for j in range(len(self.output_layer[i].weight)):  #index through the weight matrix
                for k in range(len(self.output_layer[i].weight[0])):
                    self.output_layer[i].weight[j][k] -= learning_rate * dLoss_dW_output[i][j][k]
        """
        #Now we need to backpropogate throughout the hidden matrix, starting with the last layer and going towards the first
"""

        # Backpropagate gradient throughthe hidden layers - working backwards so reverse
        for h in reversed(range(len(self.hidden_matrix))):
            for j in range(len(self.hidden_matrix[h])):
                neuron = self.hidden_matrix[h][j]
                weighted_input = np.dot(neuron.vector, neuron.weight) + neuron.bias
                gradient_activation = neuron.grad_activation(weighted_input)
          #dot product calculationz with grad activation
                delta_neuron = gradient_activation * np.sum([#i changed grad_activation to gradient activation
                    np.dot(neuron.weight[k], self.hidden_matrix[i+1][k].delta) for k in range(len(neuron.weight))
                ])

                # Update the weights/bias for the hidden neuron (we already did outerlayer)
                neuron.weight -= self.learning_rate * np.outer(neuron.vector, delta_neuron)
                neuron.bias -= self.learning_rate * delta_neuron

                # Save the delta for the next iteration (this is used in the hidden layers)
                neuron.delta = delta_neuron#idk what this is for?
    
    def backpropagation(self, predicted_value, output_vec):
        learning_rate = self.learning_rate
        # Step 1: Create delta value
        delta_output = predicted_value - output_vec

        # Step 2: Backpropagate gradient through the output layer
        Loss = self.Loss_Function(predicted_value, output_vec)
        Dloss_DA = 2 * delta_output

        # Calculate the gradient for each neuron in the output layer
        dLoss_dW_output = np.zeros((len(self.output_layer), self.weight_matrix_demensions[0], self.weight_matrix_demensions[1]))
        for i in range(len(self.output_layer)):  # For each neuron
            dLoss_dW_output[i] = np.outer(Dloss_DA[i], self.output_layer[i].vector)

            # Update the weight matrix for the current neuron
            self.output_layer[i].weight -= learning_rate * dLoss_dW_output[i]

        # Step 3: Backpropagate gradient through the hidden layers
        hidden_errors = [np.zeros_like(layer[0].weight) for layer in self.hidden_matrix]

        for i in range(len(self.output_layer)):
            Dloss_Dhidden = np.dot(self.output_layer[i].weight, Dloss_DA[i].T).T
            hidden_errors[-1] += Dloss_Dhidden

        for i in range(len(self.hidden_matrix) - 2, -1, -1):
            for j in range(len(self.hidden_matrix[i])):
                Dloss_Dhidden = np.dot(self.hidden_matrix[i + 1][j].weight, hidden_errors[i + 1].T).T
                hidden_errors[i][j] += Dloss_Dhidden

        # Step 4: Update the weights in the hidden layers
        for i in range(len(self.hidden_matrix)):
            for j in range(len(self.hidden_matrix[i])):
                self.hidden_matrix[i][j].weight -= learning_rate * hidden_errors[i][j]

        # Step 5: Backpropagate gradient through the input layer
        input_errors = [np.zeros_like(self.input_layer[0].weight) for _ in range(self.input_layer_demension)]

        for i in range(len(self.hidden_matrix[0])):
            Dloss_Dinput = np.dot(self.hidden_matrix[0][i].weight, hidden_errors[0].T).T
            input_errors[i] += Dloss_Dinput

        # Update the weights in the input layer
        for i in range(len(self.input_layer)):
            self.input_layer[i].weight -= learning_rate * input_errors[i]

    # Other methods and functions for network class here..."""

    def update_weights(self, network_output, target_output):
        # Step 1: Calculate the delta (error) for the output layer neurons
        delta_output = network_output - target_output

        # Step 2: Backpropagate the error through the hidden layers
        for layer_index in range(len(self.hidden_matrix) - 1, 0, -1):
            prev_layer_activations = []  # List to store activations from the previous layer
            for neuron in self.hidden_matrix[layer_index]:
                # Calculate the activations for each neuron in the current hidden layer
                # Assuming you have a forward propagation method that updates neuron.vector,
                # you can access it as neuron.vector or replace it with the correct attribute
                neuron_activation = neuron.vector
                prev_layer_activations.append(neuron_activation)

                # Calculate the delta (error) for the neuron in the current hidden layer
                next_layer_delta = np.dot(self.output_layer[0].weight, delta_output.T).T
                activation_derrivitave = np.zeros((neuron_activation.size, neuron_activation[0].size))
                for i in range(activation_derrivitave.size):
                    for j in range(activation_derrivitave[0].size):
                        activation_derrivitave[i][j] = neuron.activation_derivative(neuron_activation[i][j])
                delta_neuron = next_layer_delta * activation_derrivitave

                # Update the weights of the neuron in the current hidden layer
                neuron.weight -= self.learning_rate * delta_neuron * neuron_activation

            # Set the delta_output for the next iteration to be the delta of the current layer
            delta_output = next_layer_delta

        # Step 3: Update the weights for the output layer
        for neuron in self.output_layer:
            # Calculate the delta (error) for the neuron in the output layer
            delta_neuron = delta_output * neuron.activation_derivative(neuron.vector)

            # Update the weights of the neuron in the output layer
            neuron.weight -= self.learning_rate * delta_neuron * neuron.vector

        # Step 4: Update the weights for the input layer
        for neuron in self.input_layer:
            prev_layer_activations = []  # List to store activations from the previous layer
            for hidden_neuron in self.hidden_matrix[0]:
                # Calculate the activations for each neuron in the input layer
                # Assuming you have a forward propagation method that updates neuron.vector,
                # you can access it as neuron.vector or replace it with the correct attribute
                neuron_activation = neuron.vector
                prev_layer_activations.append(neuron_activation)

                # Calculate the delta (error) for the neuron in the input layer
                next_layer_delta = np.dot(hidden_neuron.weight, delta_output.T).T
                delta_neuron = next_layer_delta * neuron.activation_derivative(neuron_activation)

                # Update the weights of the neuron in the input layer
                neuron.weight -= self.learning_rate * delta_neuron * neuron_activation

           
    def train(self):
        
        epochs = len(self.train_set)
        for epoch in range(epochs):
            input_vec = self.train_set[epoch][0]
            input_tensor = input_vec.reshape((1, len(input_vec), len(input_vec[0])))
            output_vec = self.train_set[epoch][1]
            predicted_value = self.forward_propogation(input_tensor)
            weights = []
            for i in range(len(self.hidden_matrix + 2)):
                weights.append([])
                for j in range(len()
            
        print(self.forward_propogation(self.val_set[0][0]))
            
            
            
               
        
        
    
        
        
        
      
  