import DynNet


if __name__ == '__main__':
	l1 = DynNet.Layer(5)
	l2 = l1.add_layer(5, DynNet.LayerType.dense, DynNet.InitType.random)

	for neuron in l1.neurons:
		print(len(neuron.synapses))

	print('--------------')

	for neuron in l2.neurons:
		print(len(neuron.synapses))
