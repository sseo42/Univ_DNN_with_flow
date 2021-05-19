import sys, os
sys.path.append("./database")
from flow_net import Flow_net
import numpy as np
import pickle
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

# Data
(train_image_data, train_label_data), (test_image_data, test_label_data) = load_mnist(
    flatten=True, normalize=False, one_hot_label=True)

# Init Flow_net (input_size, output_size, hub_size, net_height, net_width)
test_network = Flow_net(784, 10, 10, 8, 8)

# Mini-batch variables
train_image_size = train_image_data.shape[0]
batch_size = 1000

# Train
train_acc_list = [test_network.get_accuracy(train_image_data, train_label_data)]
test_acc_list = [test_network.get_accuracy(test_image_data, test_label_data)]

## epoch 1000
epoch_cnt = 1
iter_per_epoch = max(train_image_size // batch_size, 1)
for i in range(10000000):
    batch_mask = np.random.choice(train_image_size, batch_size)
    x_batch = train_image_data[batch_mask]
    y_batch = train_label_data[batch_mask]

    test_network.update(x_batch, y_batch)

    if (i % iter_per_epoch == 0):
        train_acc = test_network.get_accuracy(train_image_data, train_label_data)
        test_acc = test_network.get_accuracy(test_image_data, test_label_data)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("epoch", epoch_cnt, "is done")
        epoch_cnt += 1
        if (epoch_cnt > 300):
            break

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

epoch_list = [i for i in range(len(train_acc_list))]
ax1.plot(epoch_list, train_acc_list, color= 'blue', linestyle= '--', label= 'train')
ax1.plot(epoch_list, test_acc_list, color= 'red', label= 'test')

variables_hist = test_network.get_hist()
print("Nan -> ", len(variables_hist[np.isnan(variables_hist)]))
print("Inf -> ", len(variables_hist[np.isinf(variables_hist)]))
ax2.hist(variables_hist[~np.isnan(variables_hist)], 
    bins= 100, density= True, alpha= 0.7, histtype= 'stepfilled')

# plt.scatter(epoch_list, train_acc_list, c= 'b')
# plt.scatter(epoch_list, test_acc_list, c= 'r')
ax1.set_xlabel("epochs")
ax1.set_ylabel("accuracy")
ax1.legend(loc= 'best')
plt.tight_layout()
plt.show()