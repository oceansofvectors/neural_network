import numpy as np
from nn import MultiClassNN
import pandas as pd

df_train = pd.read_csv('data/train.csv')

# Get the column "label" as y
y = df_train['label'].values
y = np.eye(10)[y]

# Get the other columns as X
X = df_train.drop('label', axis=1).values / 255.0  # Normalized

input_size = X.shape[1]
hidden_size = 500
output_size = 10

nn = MultiClassNN(input_size, hidden_size, output_size)
nn.train(X, y)



