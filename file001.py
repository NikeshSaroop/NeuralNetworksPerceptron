#   Perceptron Example
import numpy as np


# Sigmoid Function(maps any value to a value between 1 and 0)
# We use the sigmoid function to convert numbers into probabilities
# one of the desirable properties of the sigmoid function is that its output can be used to create its derivative.
# if the sigmoids output is variable "out" then the derivative is simply // out * (1 - out)
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


#   input Dataset
x = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1]])

#   output Dataset ".T" is used to transpose the array, example : it will split "y" into 5 different arrays.
y = np.array([[0, 0, 1, 1, 1]]).T

#   seed random numbers to make calculation
np.random.seed(1)

# initialise weights randomly with mean 0. since we only have 2 layers(input and output),we only need one matrix of
# weights to connect them.its dimension is (3,1) because we have 3 inputs and 1 output.
weights = 2 * np.random.random((3, 1)) - 1

#   Training code
#   forward propagation
for i in range(1000):
    #   "x" contains 4 training examples(rows).we're going to process all of them at the same time below.
    inputlayer = x
    #   this is our prediction step.basically we first let the network try to predict the output based on the..
    #   ..given input.
    #   1st step : the first matrix multiplies the inputlayer and the weights
    #   2nd step : passes the output to the sigmoid function
    hiddenlayer = sigmoid(np.dot(inputlayer, weights))

    #   we now compare how well it did by subtracting the true answer(y) from the guess(hiddenlayer)
    l1_error = y - hiddenlayer

    #   multiply how much we missed by the slope of the sigmoid in values 1
    l1_delta = l1_error * sigmoid(hiddenlayer, True)

    #  update weights
    weights += np.dot(inputlayer.T, l1_delta)

    print("Output After Training")
    print(hiddenlayer)
