import numpy as np

def soft_max(x_origin):
    if x_origin.ndim == 1:
        x = x_origin.reshape(1, x.size)
    else:
        x = x_origin
    c = np.max(x)
    exp = np.exp(x - c)
    exp_sum = np.sum(exp, axis= 1)
    exp = exp.T
    return (exp / exp_sum).T

def mse(y, t):
    return 0.5*np.sum((y-t)**2)

def cee(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    delta = 1e-7
    batch_size = y.shape[0]

    try:
        if y.shape[1] == t.shape[1]: # one-hot label
            return -np.sum(t*np.log(y + delta)) / batch_size
        else:
            return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
    except: # check here
        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
