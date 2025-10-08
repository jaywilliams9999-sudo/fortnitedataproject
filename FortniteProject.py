import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

main_df = pd.read_csv("Fortnite Statistics.csv")
revised_df = main_df.drop("Mental State", axis=1)
print(revised_df)