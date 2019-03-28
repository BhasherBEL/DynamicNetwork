from enum import Enum
import Activation
import Initialiser
import numpy as np


class LayerType(Enum):
	dense = 1
	free = 2


class NeuronType(Enum):
	normal = 1
	input = 2
	output = 3


class Network:

	def __init__(self, n_neurons_for_each_layer, name='unknown', activation=Activation.sigmoid, learning_rate=0.001, layer_type=LayerType.dense):
		self.layers = []
		self.n_input = n_neurons_for_each_layer[0]
		for n_neurons in n_neurons_for_each_layer[1:]:
			if len(self.layers) == 0:
				self.layers.append(Layer(n_neurons, NeuronType.input))
			else:
				self.layers.append(self.layers[-1].add_layer(n_neurons, layer_type=layer_type))

		self.name = name
		self.activation = activation
		self.learning_rate = learning_rate

	def init(self, init_type=Initialiser.random):
		for layer in self.layers:
			layer.init(init_type)

	def execute(self, inputs: list) -> list:
		if len(inputs) != self.n_input:
			raise Exception('Number of values different from the number of input neurons')
		values = [inputs]
		for layer in self.layers:
			layer_values = []
			for i in range(len(layer.neurons)):
				layer_values.append(self.activation(layer.neurons[i].execute(values[len(values)-1])))
			values.append(layer_values)

		return values

	def train(self, inputs: list, outputs: list) -> float:
		values = np.array(self.execute(inputs))

		if len(outputs) != len(values[-1]):
			raise Exception('Number of values different from the number of output neurons')

		for i in range(len(outputs)):
			values[-1][i] = outputs[i] - values[-1][i]

		error = sum(values[-1])

		for i in range(len(values)-1, 0, -1):
			for j in range(len(values[i-1])):
				for k in range(0, len(self.layers[i-1].neurons[j].weight)):
					somme = 0
					for l in range(0, len(values[i-1])):
						somme += values[i-1][l] * self.layers[i-1].neurons[j].weight[k]
					somme = self.activation(somme)

					# Error here : k is to bigger
					self.layers[i-1].neurons[j].weight[k] -= self.learning_rate * (-1 * values[i][k] * somme * (1 - somme) * values[i-1][j])

			for j in range(0, len(values[i-1])):
				somme = 0
				for k in range(0, len(values[i])):
					somme += values[i][k] * self.layers[i-1].neurons[j].weight[k]
				values[i-1][j] = somme

		return error


class Layer:

	def __init__(self, n_neurons, neuron_type=NeuronType.normal):
		self.neurons = []
		for _ in range(n_neurons):
			self.neurons.append(Neuron())
		if neuron_type == NeuronType.input:
			for neuron in self.neurons:
				neuron.weight = [None] * n_neurons

	def add_layer(self, n_neurons, layer_type=LayerType.dense):
		new_layer = Layer(n_neurons)

		if layer_type == LayerType.dense:
			for neuron in self.neurons:
				for new_neuron in new_layer.neurons:
					neuron.add_synapse(new_neuron)
		return new_layer

	def init(self, init_type=Initialiser.random):
		for neuron in self.neurons:
			neuron.init(init_type)


class Neuron:

	def __init__(self):
		self.synapses = []
		self.weight = []
		self.adder = None

	def init(self, init_type=Initialiser.random):
		self.weight = [init_type()] * len(self.weight)
		self.adder = init_type()

	def reset_synapses(self):
		self.synapses = []

	def add_synapse(self, other_neuron):
		other_neuron.weight.append(None)
		self.synapses.append(other_neuron)

	def get_synapses(self):
		return self.synapses

	def remove_synapse(self, other_neuron):
		del self.weight[self.synapses.index(other_neuron)]
		self.synapses.remove(other_neuron)

	def get_linked_neurons(self):
		return self.synapses

	def execute(self, list_x: list) -> float:
		return sum([list_x[i] * self.weight[i] for i in range(len(list_x)-1)]) + self.adder
