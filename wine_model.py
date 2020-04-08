from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
import pandas as pd


def train_model(X, y):
    m = LinearRegression()
    m.fit(X, y)
    return m


if __name__ == '__main__':
    d = load_wine()
    df = pd.DataFrame(d['data'], columns=d['feature_names'])
    y = d['target']  # cultivator

    m.score(df, y)
