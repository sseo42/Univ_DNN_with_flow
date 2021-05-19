import numpy as np
from functions import soft_max as softmax
from functions import cee

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis= 0)
        return np.dot(dout, self.W.T)

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x_origin):
        self.mask = (x_origin>=0)
        x = x_origin*self.mask
        return x

    def backward(self, dout):
        return dout*self.mask

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)

class Softmax_with_cee:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        # try:
        self.y = softmax(x)
        # except:
        #     for x_row in x:
        #         print(x_row)
        #     raise

        self.t = t
        self.loss = cee(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        tmp = dout * (self.y - self.t) / self.y.shape[0]
        for a in tmp:
            for aa in a:
                if (np.isnan(aa)):
                    exit(0)
        return tmp