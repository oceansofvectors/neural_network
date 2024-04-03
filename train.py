import numpy as np
from nn import MultiClassNN, Layer
import pandas as pd
from nn import sigmoid
from helpers import softmax

df_train = pd.read_csv('data/train.csv')

# Get the column "label" as y
y = df_train['label'].values
y = np.eye(10)[y]

# Get the other columns as X
X = df_train.drop('label', axis=1).values / 255.0  # Normalized

input_size = X.shape[1]
hidden_size1 = 128
hidden_size2 = 64
output_size = 10

layers = [
    Layer(input_size, hidden_size1, sigmoid),
    Layer(hidden_size1, hidden_size2, sigmoid),
    Layer(hidden_size2, output_size, softmax)
]

nn = MultiClassNN(layers, learning_rate=0.1)
nn.train(X, y, epochs=50, batch_size=64)
