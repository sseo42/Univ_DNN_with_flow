import layers
import numpy as np
import random

class Node:
    def __init__(self, row_size, col_size, weight_init_std = 0.1):
        self.row_size = row_size
        self.col_size = col_size
        self.visited = False
        self.path_from = None
        self.path_to = None

        #initializer
        self.Affine = layers.Affine(weight_init_std * np.random.randn(row_size, col_size), np.zeros(col_size))
        self.Relu = layers.Relu()
        self.links = [] # [node_to1, node_to2, ...]
        self.initial_links = []

    def link_node(self, node_to):
        self.initial_links.append(node_to)

    def restore_links(self):
        self.links = self.initial_links.copy()

    def find_path(self): # dfs, search order -> random choice
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
        return False

    def forward(self, x):
        if (self.path_to == None):
            raise Exception("foward: Wrong path connection")
        res = self.Relu.forward(self.Affine.forward(x))
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

    def get_variables(self):
        return self.Affine.W.flatten()

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

        return False

    def backward(self, dout = 1):
        self.Affine.backward(self.Relu.backward(dout))


class Node_end(Node):
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