from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_boston
from sklearn import tree


house = load_boston()
feature_name = house.feature_names
feature = house.data
target = house.target


model = RandomForestClassifier(n_estimators=15)
model = model.fit(feature, target.astype(int))
print(model.predict(feature))


model2 = tree.DecisionTreeRegressor()
model2.fit(feature, target.astype(int))
print(model2.predict(feature))

