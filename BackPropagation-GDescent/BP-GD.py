import numpy as np

##Methodology
#save the activation and derivaties
#TODO Implement backpropagation
#TODO Implement gradient descent
#TODO Implement  train
#TODO train our net wit some dummy database
#TODO make some predictions

class MLP(object):

    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivative = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1])) #COMEBACK idk why we are using this line
            derivative.append(d)
        self.derivative = derivative


    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.
        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs
        self.activations[0] = inputs

        # iterate through the network layers
        for i, w in enumerate(self.weights):

            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)
            self.activations[i +1] = activations

        # return output layer activation
        return activations
    
    def back_propagate(self, error, verbose = False):
        
        for i in reversed(range(len(self.derivative))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations) # ndarray([0.1, 0.2]) --> ndarray([[0.1, 0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i] # ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivative = np.dot(current_activations_reshaped,delta_reshaped) 
            error = np.dot(delta, self.weights[i].T) #i-1 error. reusing values

            if verbose:
                print("Derivatives for W{}: {}".format(i,self.derivative[i]))
        return error
    
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)


    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        
        y = 1.0 / (1 + np.exp(-x))
        return y


if __name__=="__main__":

    # create a Multilayer Perceptron
    mlp = MLP(2,[5],1)
    
    #create dummy data
    input = np.array([0.1,0.2])
    target = np.array([0.3])
    
    #forward propagation
    output = mlp.forward_propagate(input)

    #Calculate error
    error = target - output

    #bask propagation
    mlp.back_propagate(error,verbose=True)

    # print("Network activation: {}".format(output))