import DynNet
import Initialiser

if __name__ == '__main__':
	network = DynNet.Network([3, 5, 7, 3], name='test1')
	network.learning_rate = 0.05
	network.init(Initialiser.random)

	network.train([1, 0, 1], [1, 1, 0])
	network.train([1, 0, 0], [0, 0, 0])
	network.train([0, 0, 0], [1, 0, 1])
	network.train([1, 1, 1], [1, 1, 1])

