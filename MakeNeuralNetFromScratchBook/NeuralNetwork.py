import numpy as np
import scipy.special  # expit

class NeuralNetwork:

  def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1):
    self.inodes = input_nodes
    self.hnodes = hidden_nodes
    self.onodes = output_nodes

    self.learning_rate = learning_rate

    # Link Weights
    # link weight matrices, wih and who
    # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer

    # w_11, w_21 - col,row??
    # w_21, w_22 etc

    # wih - weights_input2hidden
    # who - weights_hidden2output
    # Important: Get the dimensions right
    self.weights_input2hidden = np.random.normal(
        0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
    self.weights_hidden2output = np.random.normal(
        0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html
    self.activation_function = lambda x: scipy.special.expit(x)

  def train(self, input_list, targets_list):
    inputs = np.array(input_list, ndmin=2).T
    targets = np.array(targets_list, ndmin=2).T

    # Calculate signals into hidden layer
    hidden_inputs = np.dot(self.weights_input2hidden, inputs)

    hidden_outputs = self.activation_function(hidden_inputs)
    # print(f"weights_hidden2output={self.weights_hidden2output.shape} / hidden_outputs shape={hidden_outputs.shape}")

    final_inputs = np.dot(self.weights_hidden2output, hidden_outputs)

    final_outputs = self.activation_function(final_inputs)

    # output layer error is the (target ­ actual)
    output_errors = targets - final_outputs

    # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
    hidden_errors = np.dot(self.weights_hidden2output.T, output_errors)

    # update the weights for the links between the hidden and output layers
    self.weights_hidden2output += self.learning_rate * \
        np.dot((output_errors * final_outputs * (1 - final_outputs)),
               np.transpose(hidden_outputs))

    self.weights_input2hidden += self.learning_rate * \
        np.dot((hidden_errors * hidden_outputs *
               (1 - hidden_outputs)), np.transpose(inputs))

  def query(self, input_list):
    # convert inputs list to 2d numpy array
    inputs = np.array(input_list, ndmin=2).T

    # Calculate signals into hidden layer
    hidden_inputs = np.dot(self.weights_input2hidden, inputs)

    hidden_outputs = self.activation_function(hidden_inputs)

    final_inputs = np.dot(self.weights_hidden2output, hidden_outputs)

    final_outputs = self.activation_function(final_inputs)

    return final_outputs
