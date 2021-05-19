from node import *
import layers
import numpy as np
import random

class Flow_net:
    def __init__(self, input_size, output_size, hub_size, net_height, net_width):
        self.hub_size = hub_size
        self.net_height = net_height
        self.net_width = net_width
        self.start_node = Node_start(input_size, self.hub_size)
        self.hidden_hubs = [[Node(self.hub_size, self.hub_size) for _ in range(
            self.net_width)] for i in range(self.net_height)]
        self.last_node = Node_end(self.hub_size, output_size)
        self.label_activation = layers.Softmax_with_cee()

        #connection model
        self.init_full_connection_between_layers()

    def predict(self, x):
        self.reset_links()
        self.reset_visit()
        if (self.start_node.find_path() == False):
            raise Exception("predict: No path")
        res = self.start_node.forward(x)
        return res

    def get_loss(self, x, t):
        res = self.predict(x)
        return self.label_activation.forward(res, t)

    def get_accuracy(self, x, t):
        predicted = np.argmax(self.predict(x), axis= 1)
        if (t.ndim != 1):
            t = np.argmax(t, axis= 1)
        return np.sum(predicted == t) / float(x.shape[0])

    def update(self, x, t, learning_rate = 0.01):
        self.reset_links()
        self.reset_visit()

        while (self.start_node.find_path()):
            res = self.start_node.forward(x)
            loss = self.label_activation.forward(res, t)
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

    #Initially connected model
    def init_full_connection_between_layers(self):
        for i in range(self.net_height):
            self.start_node.link_node(self.hidden_hubs[i][0])
        for i in range(self.net_width - 1):
            for j in range(self.net_height):
                for k in range(self.net_height):
                    self.hidden_hubs[j][i].link_node(self.hidden_hubs[k][i + 1])
        for i in range(self.net_height):
            self.hidden_hubs[i][self.net_width - 1].link_node(self.last_node)

    def init_model(self):
        print("yet")
        # self.start_node.link_node(self.last_node)
        # for idx in range(1,6):
        #     for _ in range(idx * idx):
        #         random.choice(random.choice([[self.start_node]] + self.hidden_hubs[0:idx])).link_node(
        #             random.choice(random.choice(self.hidden_hubs[(10 - idx)::] + [[self.last_node]])))

    def get_hist(self):
        tmp = np.concatenate((self.start_node.get_variables(), self.last_node.get_variables()), axis= None)
        print(tmp)
        for layer in self.hidden_hubs:
            for node in layer:
                tmp = np.concatenate((tmp, node.get_variables()), axis= None)
        return tmp