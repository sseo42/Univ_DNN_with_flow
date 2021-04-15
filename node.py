import numpy as np
import random
import layers

class Node:
    def __init__(self, row_size, col_size):
        self.row_size = row_size
        self.col_size = col_size
        self.Affine = layers.Affine(np.random.randn(row_size, col_size), np.zeros(col_size))
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
        random.shuffle(self.links)
        for idx in range(len(self.links)):
            if (self.links[idx].visited == False):
                self.links[idx].path_from = self
                if (self.links[idx].find_path()):
                    self.path_to = self.links[idx]
                    self.links.pop(idx)
                    self.links.append(self.path_from)
                    return True
        self.visited = False
        return False

    def forward(self, x):
        if (self.path_to == None):
            raise Exception("foward: Wrong path connection")
        res = self.Relu.forward(self.Affine.forward(x))
        print("dang!")
        return self.path_to.forward(res)

    def backward(self, dout = 1):
        if (self.path_from == None):
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
        random.shuffle(self.links)
        for idx in range(len(self.links)):
            if (self.links[idx].visited == False):
                self.links[idx].path_from = self
                if (self.links[idx].find_path()):
                    self.path_to  = self.links[idx]
                    self.links.pop(idx)
                    return True
        self.visited = False
        return False

    def backward(self, dout = 1):
        self.Affine.backward(self.Relu.backward(dout))


class Node_end(Node):
    def __init__(self, row_size, col_size):
        self.Affine = layers.Affine(np.random.randn(row_size, col_size), np.zeros(col_size))

        self.links = []
        self.initial_links = []

        self.visited = False
        self.path_from = None
        self.path_to = None
        
    def find_path(self):
        return True

    def forward(self, x):
        res = self.Affine.forward(x)
        return res
    
    def backward(self, dout = 1):
        if (self.path_from == None):
            raise Exception("backward: Wrong path connection")
        res = self.Affine.backward(dout)
        self.path_from.backward(res)

class Flow_net:
    def __init__(self, input_size, output_size):
        hub_size = 20

        self.start_node = Node_start(input_size, hub_size)
        self.hidden_hubs = [[Node(hub_size,hub_size) for _ in range(10)] for i in range(10)]
        self.last_node = Node_end(hub_size, output_size)

        self.start_node.link_node(self.last_node)
        for idx in range(1,6):
            for _ in range(idx * idx):
                random.choice(random.choice([[self.start_node]] + self.hidden_hubs[0:idx])).link_node(
                    random.choice(random.choice(self.hidden_hubs[10 - idx::] + [[self.last_node]])))

        print("test!!! ", self.start_node.initial_links)
        self.label_activation = layers.Softmax_with_cee()

    def predict(self, x): # check here
        self.reset_links()
        self.reset_visit()

        res = self.start_node.forward(x)
        return soft_max(res)

    def get_loss(self, x, t):
        res = self.predict(x)
        return self.label_activation.forward(res, t)

    def get_accuracy(self, x, t):
        print("not yet")

    def update(self, x, t, learning_rate = 0.01):
        self.reset_links()
        self.reset_visit()

        while (self.start_node.find_path()):
            res = self.start_node.forward(x)
            print("forward end!")
            loss = self.label_activation.forward(res, t)
            print("test y: ", self.label_activation.y)
            print("test loss", loss)
            self.last_node.backward(self.label_activation.backward())

            target_node = self.start_node
            while (target_node != None):
                target_node.Affine.W -= learning_rate * target_node.Affine.dw
                target_node.Affine.b -= learning_rate * target_node.Affine.db
                target_node = target_node.get_path_to()
            
            self.reset_visit()

    def reset_visit(self):
        for layer in self.hidden_hubs:
            for node in layer:
                node.visited = False
        self.start_node.visited = False
        self.last_node.visited = False

    def reset_links(self):
        for layer in self.hidden_hubs:
            for node in layer:
                node.restore_links()
        self.start_node.restore_links()
        self.last_node.restore_links()