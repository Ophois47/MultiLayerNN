# To ignore numpy errors:
#   pylint: disable=E1101

from numpy import exp, array, random, dot

class NeuronLayer():
    def __init__(self, neurons_amount, inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((inputs_per_neuron, neurons_amount)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # Sigmoid Function Passes Weighted Sum Of Inputs Through Itself To
    # Normalise Them Between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Derivative Of Sigmoid, Gradient Of Sigmoid Curve.
    # Indicates Confidence In Existing Weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Using Trial + Error, Train Network
    # Adjusting Synaptic Weights Each Time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass Training Set Through.
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate Error For Layer 2 (Difference Between Desired 
			# Output + Predicted Output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate Error For Layer 1. Weights In Layer 1
            # Determined By How Much Layer 1 Contributed To Error In Layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate How Much To Adjust Weights By.
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust Weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The Network Thinks Here.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # Print The Network Weights
    def print_weights(self):
        print("    Layer 1 (4 Neurons, Each With 3 Inputs):")
        print(self.layer1.synaptic_weights)
        print("    Layer 2 (1 Neuron, With 4 Inputs):")
        print(self.layer2.synaptic_weights)

if __name__ == "__main__":

    #Seed Random Number Generator.
    random.seed(1)

    # Create Layer 1 (4 Neurons, Each 3 Inputs)
    layer1 = NeuronLayer(4, 3)

    # Create Layer 2 (Single Neuron, 4 Inputs)
    layer2 = NeuronLayer(1, 4)

    # Combine Layers To Create Neural Network.
    neural_network = NeuralNetwork(layer1, layer2)

    print("Stage 1) Random Starting Synaptic Weights: ")
    neural_network.print_weights()

    # Create Initial Training Set. I'm Using 7 Examples, 
	# Each Consisting Of 3 Input Values + 1 Output Value.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train Network Using Above Training Set.
    # 60,000 Iterations While Making Small Adjustments Each Time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("Stage 2) New Synaptic Weights After Training: ")
    neural_network.print_weights()

    # Test Network With A New Situation.
    print("Stage 3) Considering A New Situation [1, 1, 0] -> ?: ")
    hidden_state, output = neural_network.think(array([1, 1, 0]))
    print(output)