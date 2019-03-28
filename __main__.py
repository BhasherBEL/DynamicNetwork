import DynNet


if __name__ == '__main__':
	network = DynNet.Network([5, 10, 20, 10, 5, 1], name='test1')

	for layer in network.layers:
		print('----------')
		for neuron in layer.neurons:
			print(len(neuron.synapses))
