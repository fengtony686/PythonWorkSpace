import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

train_feature = np.genfromtxt("train_feat.txt", dtype=np.float32)
num_feature = len(train_feature[0])
train_feature = pd.DataFrame(train_feature)
train_label = train_feature.iloc[:, num_feature - 1]
train_feature = train_feature.iloc[:, 0:num_feature-2]


test_feature = np.genfromtxt("test_feat.txt", dtype=np.float32)
test_feature = pd.DataFrame(test_feature)
test_label = test_feature.iloc[:, num_feature-1]
test_feature = test_feature.iloc[:, 0:num_feature-2]


gbdt = GradientBoostingClassifier(
    loss="deviance",
    learning_rate=0.1,
    n_estimators=100,
    subsample=1,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,
    init=None,
    random_state=None,
    max_features=None,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False
)


gbdt.fit(train_feature, train_label)
pred = gbdt.predict(test_feature)
total_err = 0
for i in range(pred.shape[0]):
    print('pred:', pred[i], 'label', test_label[i])
print('MSE:', np.sqrt((pred - test_label) ** 2))

