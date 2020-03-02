import pandas as pd
from urllib.request import urlretrieve


def load_data(download=True):
    if download:
        data_path, _ = urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
                                   "car.csv")
        print("Downloaded to car.csv")

    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    data = pd.read_csv("car.csv", names=col_names)
    return data


def convert2onehot(data):
    return pd.get_dummies(data, prefix=data.columns)