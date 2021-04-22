import numpy as np

#Multi-Layer Perceptions
class MLP:
    def __init__(self, num_imputs=3, hidden_layers=[3,5],num_outputs=2):
        self.num_imputs = num_imputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [self.num_imputs] + self.hidden_layers + [self.num_outputs]

        #initiate random layers weight
        self.weight = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weight.append(w)
    
    def fordware_propagate(self,inputs):
        activations = inputs

        for w in self.weight:
            #calcilate net inputs
            net_input = np.dot(activations,w)

            #calculate the activations (Sigmoid activation function)
            activations  = self._sigmoid(net_input)

        return activations
    
    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    


if __name__ == "__main__":
    
    #create an MLP
    mlp = MLP()
    
    #Create some inputs
    inputs = np.random.rand(mlp.num_imputs)
    
    #performace fordware_propagation
    outputs = mlp.fordware_propagate(inputs)
    
    #print the results
    print(f"The network inputs is: {inputs}")
    print(f"The network outputs is: {outputs}")

