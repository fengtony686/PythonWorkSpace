import numpy as np
import matplotlib.pyplot as plt
import DataLoading
import matplotlib
import torch.nn as nn
import torch
import torch.nn.functional as F

LR = 0.1
EPOCH = 4000
matplotlib.use(backend='TkAgg')


data = DataLoading.load_data(download=True)
new_data = DataLoading.convert2onehot(data)
new_data = new_data.values.astype(np.float32)
np.random.shuffle(new_data)
sep = int(0.7*len(new_data))
train_data = new_data[:sep]
test_data = new_data[sep:]
print(train_data[1])
X_train = train_data[:, :21]
Y_train = train_data[:, 21:]
X_test, Y_test = test_data[:, :21], test_data[:, 21:]


class Net(nn.Module):
    def __init__(self, n_features, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 128, nn.ReLU())
        self.fc2 = nn.Linear(128, 128, nn.ReLU())
        self.out = nn.Linear(128, n_output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


net = Net(21, 4)
print(net)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for i in range(sep):
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        output = net(X_train[i])
        print(X_train[i])
        print(Y_train[i])
        print(output)
        loss = loss_func(output, Y_train[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            X_test = torch.from_numpy(X_test)
            pred_y = net(X_test).numpy()
            pred_y = np.argmax(pred_y, 1)
            Y_test = np.argmax(Y_test, 1)
            plt.scatter(sep, Y_test)
            plt.scatter(sep, pred_y)
            plt.show()