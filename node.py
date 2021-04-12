import numpy as np
import layers

class Node:
    def __init__(self, row_size, col_size):
        self.Affine = layers.Affine(np.ones((row_size, col_size)), np.zeros(col_size))
        self.Relu = layers.Relu()

        self.links = [] # [node_to1, node_to2, ...]
        self.initial_links = []

        self.visited = False
        self.path_from = None
        self.path_to = None

    def link_node(self, node_to):
        self.initial_links.append(node_to)

    def restore_links(self):
        self.links = self.initial_links.copy()

    def find_path(self):
        self.visited = True
        for idx in range(len(self.links)): # list 자료구조 바꿀것
            links[idx].path_from = self
            if (links[idx].find_path()):
                self.path_to = links[idx]
                links.pop(idx)
                links.append(self.path_from)
                return True

        self.visited = False
        return False

    def forward(self, x):
        if (self.path_to == None)
            raise Exception("foward: Wrong path connection")
        res = self.Relu.forward(self.Affine.forward(x))
        return self.path_to.forward(res)

    def backward(self, dout = 1):
        if (self.path_from == None)
            raise Exception("backward: Wrong path connection")
        res = self.Affine.backward(self.Relu.backward(dout))
        self.path_from.backward(res)

    def get_path_to(self):
        return self.path_to

    def get_path_from(self):
        return self.path_from


class Node_start(Node):
    def find_path(self):
        self.visited = True
        for idx in range(len(self.links)):
            links[idx].path_from = self
            if (links[idx].find_path()):
                self.path_to  = links[idx]
                links.pop(idx)
                return True
        self.visited = False
        return False

    def backward(self, dout = 1):
        self.Affine.backward(self.Relu.backward(dout))


class Node_end(Node):
    def __init__(self, row_size, col_size):
        self.Affine = layers.Affine(np.ones((row_size, col_size)), np.zeros(col_size))

        self.links = [] # [node_to1, node_to2, ...]
        self.initial_links = []

        self.visited = False
        self.path_from = None
        self.path_to = None
        
    def find_path(self):
        return True

    def forward(self, x):
        res = self.Affine(x)
        return res
    
    def backward(self, dout = 1):
        if (self.path_from == None)
            raise Exception("backward: Wrong path connection")
        res = self.Affine.backward(dout)
        self.path_from.backward(res)

class Flow_net:
    def __init__(self):
        self.start_node = Node_start(3, 7)

        first = [Node(7,2), Node(7, 6)]

        second = [Node(2,3), Node(2,4), Node(6,3), Node(6,4)]

        third = [Node(3,5), Node(4,5)]

        self.last_node = Node_end(5,2)

        start_node.link_node(first[0])
        start_node.link_node(first[1])

        first[0].link_node(second[0], second[1])
        first[1].link_node(second[2], second[3])

        second[0].link_node(third[0])
        second[1].link_node(third[0])
        second[2].link_node(third[1])
        second[3].link_node(third[1])

        third[0].link_node(last_node)
        third[1].link_node(last_node)

        self.all_nodes = [self.start_node, self.last_node] + first + second + third

        self.label_activation = layers.Softmax_with_cee()

    def predict(self, x): # check here
        for node in self.all_nodes:
            node.restore_links()
            node.visited = False
        res = self.start_node.forward(x)
        return soft_max(res)

    def get_loss(self, x, t):
        res = self.predict(x)
        return self.label_activation.forward(res, t)

    def get_accuracy(self, x, t):

    def update(self, x, t, learning_rate = 0.01):
        for node in self.all_nodes:
            node.restore_links()
            node.visited = False

        while (self.start_node.find_path()):
            res = self.start_node.forward(x)
            loss = self.label_activation(res, t)
            self.last_node.backward(loss)

            target_node = self.start_node
            while (target_node != None):
                target_node.Affine.W -= learning_rate * target_node.Affine.dw
                target_node.Affine.b -= learning_rate * target_node.Affine.db
                target_node = target_node.get_path_to
            
            for node in self.all_nodes:
                node.visited = False