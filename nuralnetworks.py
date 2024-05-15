import numpy as np

class Neuron:
    def __init__(self, layer, deminsions):
        
        self.layer = layer
        self.matrix = np.ones((deminsions, deminsions))
        self.vector = np.zeros((1, deminsions))
        self.bias = np.random.randint(1,)
        self.input_vector = np.zeros((1, deminsions))
    
    def vectoradd(self, X, Y):
        print(f'X: {X}, Y:{Y}')
        result = np.zeros((3, 1))
        
        for i in range(len(X)):  
        # iterate through columns
            for j in range(len(X[0])):
                result[i][j] = X[i][j] + Y[i][j]
        
        return result
    def matrixcoef(self, coef):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                self.matrix[i][j] = self.matrix[i][j] * coef
                
    
                
                
        
    def propogate(self, inputvec1, inputvec2, inputvec3, inputvec4):
        #multiply matrices
      

        
        vec1 = self.vectoradd(inputvec1, inputvec2)
        print(f' vec1: {vec1}')
        vec2 = self.vectoradd(inputvec3, inputvec4)
        vec = self.vectoradd(vec1, vec2)
        self.input_vector = vec
    
                

   
        
        self.vector = np.dot(self.matrix, vec)
        
        for i in range(len(self.vector)):
            self.vector[i][0] += self.bias
        for i in range(len(self.vector)):
            self.vector[i][0] = self.sigmoid(self.vector[i][0])
    
    def getVector(self):
        return self.vector
    def getWeight(self):
        return self.matrix
    def getbias(self):
        return self.bias
    def sigmoid(self, value):
        return (1 + (np.power(np.e, value * -1)))
    def get_inputvec(self):
        return self.vector
class Network:
    def __init__(self, inputvecs, dataset, dim):
        self.network = np.array([[Neuron(1, dim), Neuron(1, dim), Neuron(1, dim), Neuron(1, dim)], 
                            [Neuron(2, dim),Neuron(2, dim), Neuron(2, dim), Neuron(2, dim)],
                            [Neuron(3, dim), Neuron(3, dim), Neuron(3, dim), Neuron(3, dim)],
                            [Neuron(4, dim),Neuron(4, dim),Neuron(4, dim),Neuron(4,dim)]
                            ])
        self.neuron_matrix_deminsions = dim
        self.inputvecs = inputvecs
        self.dataset = dataset
        self.input_tensor = np.array([self.inputvecs[self.index], np.zeros((len(self.inputvec[self.index]), len(self.inputvecs[self.index][0])))])
        self.output_tensor = np.array([self.inputvecs[0], self.dataset])
        self.outputvec = np.zeros((1, dim))
        self.gradient_weight = np.zeros(len(self.network) * len(self.network[0]) * self.neuron_matrix_deminsions**2)
        self.gradient_bias = np.zeros(len(self.network) * len*self.network[0])
        self.learning_rate = 0.1
        self.index = 0
       
        
    def definite_integral(self,coefficients, x_values):
        """
        This function takes a list of coefficients and a list of x values and returns the definite integral of the polynomial function from the lowest x value to the highest x value.

        Args:
            coefficients: A list of coefficients.
            x_values: A list of x values.

        Returns:
            The definite integral of the polynomial function.
        """

        lower_x = x_values[0]
        upper_x = x_values[-1] 

        integral = 0
        for i in range(len(coefficients)):
            integral += coefficients[i] * (upper_x ** (i + 1) - lower_x ** (i + 1)) / (i + 1)

        return integral

    def lossfunction(self, nn_out):
        self.input_tensor[1] = nn_out
        x_values = np.array([])
        y_values = np.array([])
        
        integrals = np.array([[], []])
        for l in range(len(self.input_tensor[0][0])):
            for i in range(len(self.input_tensor[0])):
                x_values.append(self.input_tensor[0][i][l])
            for i in range(len(self.input_tensor[1])):
                y_values.append(self.input_tensor[1][i][l])
                
        coefficients = self.polynomial_regression(x_values, y_values_o)
        
        
        integrals[0].append(self.definite_integral(x_values, coefficients))
        
        x_values_o = np.array([])
        y_values_o = np.array([])
        for l in range(len(self.output_tensor[0][0])):
            for i in range(len(self.output_tensor[0])):
                x_values_o.append(self.output_tensor[0][i][l])
            for i in range(len(self.output_tensor[1])):
                y_values_o.append(self.output_tensor[1][i][l])
                
        coefficients_o = self.polynomial_regression(x_values_o, y_values_o)

        integrals[1].append(self.definite_integral(x_values_o, coefficients_o))
        
        


                
        
    
        
            
        

    def sigmoid(self, value):
        return 1 (1 + (np.power(np.e, value * -1)))
    def polynomial_regression(x_values, y_values):  
        coefficients = []
        for i in range(len(x_values) + 1):
            coefficients.append(0)

        for i in range(len(x_values)):
            for j in range(len(x_values) - i):
                coefficients[i + j] += y_values[i] * x_values[j]

        return coefficients
    def weighted_sum(self, inputvec, i, j):
        return np.dot(inputvec, self.network[i][j].getWeight()) + self.network[i][j].getbias()
    
    def calculate_gradient_weight(self):
        p = 0
        for i in range(len(self.network)):
            for j in range(len(self.network[0])):
                #find partial derrivitave for entire neuron
                vec = self.network[i][j].getVector()
                inputvec = self.network[i][j].get_inputvec()
                weighted_sum = self.weighted_sum(inputvec, i, j)
                sigmoid_partial_derrivitave = self.network[i][j].sigmoid(weighted_sum) * ( 1 - self.network[i][j].sigmoid(weighted_sum))
                #now that we have the partial derrivitave, the gradient for an element of a sum is that derrivitave times the next value
                for l in range(len(self.network[i][j].getweight())):
                    for m in range(len(self.network[i][j].getweight()[0])):
                        if p < len(self.gradient_weight):
                            if m < len(self.network[i][j].getweight()[0] - 1): #if its not in the last collumn, then weight is sigmoid partial derrivitave times the element in the collumn one over
                                self.gradient_weight[p] = self.sigmoid_partial_derrivitave * self.network[i][j].getweight()[l][m + 1]
                                p += 1
                            elif l == len(self.network[i][j].getweight) and m == len(self.network[i][j].getweight()[0]):# if its the last element, then its sigmoid partial derrivitave times the first element
                                self.gradient_weight[p] = self.sigmoid_partial_derrivitave * self.network[i][j].getweight()[0][0]
                                p += 1
                            else:
                                self.gradient_weight[p] = self.sigmoid_partial_derrivitave *self.network[i][j].getweight()[l + 1][0] #anything else(which is only things in the last collumn) then its sigmoid partial derrivitave times elemt in the 0th collumn and one row down
                                #incrament p
                                p += 1
                        else:
                            break# if p is more then the number of slots in the gradient matrix then stop the loop
                        
                
                
    def calculate_gradient_bias(self):
        p = 0
        for i in range(len(self.network)):
            for j in range(len(self.network[0])):
                #find partial derrivitave for entire neuron
                vec = self.network[i][j].getVector()
                inputvec = self.network[i][j].get_inputvec()
                weighted_sum = self.weighted_sum(inputvec, i, j )
                sigmoid_partial_derrivitave = self.network[i][j].sigmoid(weighted_sum) * ( 1 - self.network[i][j].sigmoid(weighted_sum))
                if p < self.gradient_bias:
                    if j < len(self.network[0] - 1):
                        self.gradient_bias[p] = sigmoid_partial_derrivitave * self.network[i][j + 1].getbias()
                        p += 1
                    elif i == len(self.network) and j == len(self.network[0]):
                        self.gradient_bias[p] = sigmoid_partial_derrivitave * self.network[0][0].getbias()
                        p += 1
                    
                    else:
                        self.gradient_bias[p] = sigmoid_partial_derrivitave *self.network[0][j + 1].getbias()
                        p += 1
    def update_weights(self):
        # index through each neuron
        p = 0
        for i in range(len(self.network)):
            for j in range(len(self.network)):
                for k in range(len(self.network[i][j].getWeight())):
                    for l in range(len(self.network[i][j].getWeight()[0])):
                        self.network[i][j].matrix[k][l] -= self.gradient_weight[p] * self.learning_rate
                        p += 1
                
    def update_biases(self):
        p = 0
        for i in range(len(self.network)):
            for j in range(len(self.network[0])):
                self.network[i][j].bias -= self.gradient_bias[p] * self.learning_rate
                p += 1
        
    def train(self):
        output = self.outputvec
        theta = 1
        i = 0
        j = 0
        self.output_vec = self.forward_propogate()
        self.gradient_weight = self.calculate_gradient_weight()
        loss = self.lossfunction(self.output_vec)
        sum_loss = np.sum(loss)
        while sum_loss > 10:
            self.gradient_weight = self.calculate_gradient_weight()
            self.gradient_bias = self.calculate_gradient_bias()
            self.update_weights()
            self.update_biases
            loss = self.lossfunction(self.output_vec)
            sum_loss = np.sum(loss)
            self.index += 1
            self.inputvec = self.inputvecs[i]
            self.forward_propogate()
            
        
       
            
            
                            
                            
            
            
                
            
            
            
            
            
        
        
        
    def forward_propogate(self):
        for i in range(len(self.network[0])):
            self.network[0][i].propogate(self.inputvec, self.inputvec, self.inputvec, self.inputvec)
        for i in range (1, len( self.network)):
            
            for j in range(len(self.network[0])):
                if self.network[i][j] != None:
                    self.network[i][j].propogate(self.network[i-1][0].getVector(), self.network[i - 1][1].getVector(), self.network[i - 1][2].getVector(),self.network[i - 1][3].getVector())
                    print(f'propogating neuron {i}, {j} with neurons: {0} {i - 1}, {1} {i - 1}, {2} {i - 1}, {3} {i - 1}')
                    if i == 4:
                        break
        self.outputvec = self.network[3][0].vector
        
        return self.outputvec
        
N = Network([[1], [1], [1]], [[8], [7],[3]], 3)
N.train()
                
            
                
            
            
            
        
        
        

        
