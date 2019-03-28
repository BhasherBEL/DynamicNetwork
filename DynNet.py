from enum import Enum
import Activation
import Initialiser


class LayerType(Enum):
	dense = 1
	free = 2


class Network:

	def __init__(self, n_neurons_for_each_layer, name='unknown', activation=Activation.sigmoid, learning_rate=0.001, layer_type=LayerType.dense):
		self.layers = []
		for n_neurons in n_neurons_for_each_layer:
			if len(self.layers) == 0:
				self.layers.append(Layer(n_neurons))
			else:
				self.layers.append(self.layers[-1].add_layer(n_neurons, layer_type=layer_type))

		self.name = name
		self.activation = activation
		self.learning_rate = learning_rate

	def init(self, init_type=Initialiser.random):
		for layer in self.layers:
			layer.init(init_type)


class Layer:

	def __init__(self, n_neurons):
		self.neurons = []
		for _ in range(n_neurons):
			self.neurons.append(Neuron())

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

	def init(self, init_type=Initialiser.random):
		for i in range(len(self.synapses)):
			self.synapses[i][1] = init_type()
			self.synapses[i][2] = init_type()

	def reset(self):
		self.synapses = []

	def add_synapse(self, other_neuron):
		self.synapses.append([other_neuron, 0, 0])

	def get_synapses(self):
		return self.synapses
