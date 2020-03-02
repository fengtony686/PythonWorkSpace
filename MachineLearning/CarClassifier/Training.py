import numpy as np
import matplotlib.pyplot as plt
import DataLoading
import matplotlib
import torch.nn as nn
import torch
import torch.nn.functional as F

LR = 0.1
EPOCH = 10
matplotlib.use(backend='WxAgg')


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
X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)
X_test = torch.from_numpy(X_test)


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
        output = net(X_train[i])
        target = np.argmax(Y_train[i].numpy())
        target = torch.from_numpy(np.array([target]))
        output = output.view(1, 4)
        loss = loss_func(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            with torch.no_grad():
                pred_y = net(X_test).numpy()
                print(pred_y[:10])
                print(Y_test[:10])


Pred_y = np.argmax(pred_y[:40], axis=1)
Y_Test = np.argmax(Y_test[:40], axis=1)
print(Pred_y)
print(Y_Test)
plt.scatter(np.arange(len(Pred_y)), Pred_y)
plt.scatter(np.arange(len(Y_Test)), Y_Test)
plt.show()