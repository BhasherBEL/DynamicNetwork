import random
from enum import Enum


class LayerType(Enum):
	dense = 1
	not_connected = 2


class InitType(Enum):
	random = 1
	zero = 2


class Layer:

	def __init__(self, n_neurons):
		self.neurons = []
		for _ in range(n_neurons):
			self.neurons.append(Neuron())

	def add_layer(self, n_neurons, layer_type, init_type):
		new_layer = Layer(n_neurons)

		if layer_type == LayerType.dense:
			for neuron in self.neurons:
				for new_neuron in new_layer.neurons:
					neuron.add_synapse(new_neuron)

				if init_type == InitType.random:
					neuron.init_with_random_values()
				elif init_type == InitType.zero:
					neuron.init_with_zero()
		return new_layer


class Neuron:

	def __init__(self):
		self.synapses = []

	def init_with_random_values(self):
		for i in range(len(self.synapses)):
			self.synapses[i][1] = random.random()
			self.synapses[i][2] = random.random()

	def init_with_zero(self):
		self.init_with_value(0, 0)

	def init_with_value(self, weight, sous):
		for i in range(len(self.synapses)):
			self.synapses[i][1] = weight
			self.synapses[i][2] = sous

	def reset(self):
		self.synapses = []

	def add_synapse(self, other_neuron):
		self.synapses.append([other_neuron, 0, 0])

	def get_synapses(self):
		return self.synapses
