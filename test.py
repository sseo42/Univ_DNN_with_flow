import node
import numpy as np

test_network = node.Flow_net()

test_x = np.random.randn(3, 3)
test_y = np.zeros((3, 2))
for i in test_y:
    i[1] = 1
test_network.update(test_x, test_y)