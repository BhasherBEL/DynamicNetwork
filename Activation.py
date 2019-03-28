import math


def sigmoid(x):
	return 1/(1+math.exp(-x))


def tangent(x):
	return 1.7159 * math.tanh((2/3)*x)
