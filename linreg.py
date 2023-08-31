import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\noahg\Desktop\School\code\Fun\Python\new\datasci\data_for_lr.csv') # read data file
data = data.dropna() # drop missing vals
train_input = np.array(data.x[0:500]).reshape(500,1) # first 500 x vals --> 500x1 np array for training
train_output  = np.array(data.y[0:500]).reshape(500,1) # first 500 y vals --> 500x1 np array for training
test_input = np.array(data.x[500:700]).reshape(199,1) # last 199 test (200 --> dropped 1)
test_output  = np.array(data.y[500:700]).reshape(199,1) # last 199 test (200 --> dropped 1)

class LinearRegression:
	def __init__(self):
		self.parameters = {}
	
	def forward_propagation(self, train_input): # m*DATA + c (linear equation)
		m = self.parameters['m']
		c = self.parameters['c']
		predictions = np.multiply(m, train_input) + c
		return predictions

	def cost_function(self, predictions, train_output): # MSE
		cost = np.mean((train_output - predictions) ** 2)
		return cost

	def backward_propagation(self, train_input, train_output, predictions): # derive wrt intercept(c) and coefficient(m)
		derivatives = {}
		df = (train_output - predictions) * -1
		dm = np.mean(np.multiply(train_input, df))
		dc = np.mean(df)
		derivatives['dm'] = dm
		derivatives['dc'] = dc
		return derivatives

	def update_parameters(self, derivatives, learning_rate):
		self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
		self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']

	def train(self, train_input, train_output, learning_rate, iters):
		self.parameters['m'] = np.random.uniform(0,1) * -1
		self.parameters['c'] = np.random.uniform(0,1) * -1
		self.loss = []
		for i in range(iters):
			predictions = self.forward_propagation(train_input)
			cost = self.cost_function(predictions, train_output)
			self.loss.append(cost)
			print("Iteration = {}, Loss = {}".format(i+1, cost))
			derivatives = self.backward_propagation(train_input, train_output, predictions)
			self.update_parameters(derivatives, learning_rate)
		return self.parameters, self.loss
	
linear_reg = LinearRegression()
parameters, loss = linear_reg.train(train_input, train_output, 0.0001, 20)

y_pred = test_input*parameters['m'] + parameters['c'] # final pred vals
 
plt.plot(test_input, test_output, '+', label='Actual values')
plt.plot(test_input, y_pred, label='Predicted values')
plt.xlabel('Test input')
plt.ylabel('Test Output or Predicted output')
plt.legend()
plt.show()