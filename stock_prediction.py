import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Load nyse_data
nyse_data = pd.read_csv("https://raw.githubusercontent.com/prathameshk03/NYSE-Stock-Prediction-Using-RNN/master/prices-split-adjusted.csv", index_col=0)
my_stock = 'AAPL'
nyse_data = nyse_data[nyse_data.symbol == my_stock]
nyse_data.drop(['symbol', 'volume'], axis=1, inplace=True)
print(nyse_data)

# Spliting the data into training part and testing part
training_part = nyse_data.iloc[:1000].values
testing_part = nyse_data.iloc[1000:].values

# Scaling the data
scaler_func = MinMaxScaler()
scaled_training_part = scaler_func.fit_transform(training_part)
scaled_testing_part = scaler_func.transform(testing_part)

# Defining sequence length
seq_length = 20

# Preparing the training part
trainPart_X = []
trainPart_Y = []
for i in range(len(training_part) - seq_length):
    trainPart_X.append(scaled_training_part[i:i + seq_length])
    trainPart_Y.append(scaled_training_part[i + seq_length, 1])  # Using the closing price at seq_length as target

trainPart_X = np.array(trainPart_X)
trainPart_Y = np.array(trainPart_Y)

# Preparing the testing part
testPart_X = []
testPart_Y = []
for i in range(len(testing_part) - seq_length):
    testPart_X.append(scaled_testing_part[i:i + seq_length])
    testPart_Y.append(scaled_testing_part[i + seq_length, 1])  # Using the closing price at seq_length as target

testPart_X = np.array(testPart_X)
testPart_Y = np.array(testPart_Y)

class RNN:
    def __init__(self, input_lr_size, hidden_lr_size, output_lr_size, learningRate=0.01):
        # Initializing the Weights and Biases
        self.hidden_lr_size = hidden_lr_size
        self.weights_XH = np.random.randn(input_lr_size, hidden_lr_size) * learningRate
        self.weights_HH = np.random.randn(hidden_lr_size, hidden_lr_size) * learningRate
        self.bias_h = np.zeros((1, hidden_lr_size))
        self.weights_HO = np.random.randn(hidden_lr_size, output_lr_size) * learningRate
        self.bias_O = np.zeros((1, output_lr_size))
        self.learningRate = learningRate

    def forward(self, x):
        # Initializing the hidden state
        h = np.zeros((1, self.hidden_lr_size))
        self.hidden_states = []
        self.outputs = []
        # Looping through the sequence
        for t in range(x.shape[0]):
            # Computing the new hidden state
            h = np.tanh(np.dot(x[t].reshape(1, -1), self.weights_XH) + np.dot(h, self.weights_HH) + self.bias_h)
            # Computing the output
            o = np.dot(h, self.weights_HO) + self.bias_O
            # Storing the hidden state and output
            self.hidden_states.append(h)
            self.outputs.append(o)
        return self.outputs[-1]

    def backward(self, x, y):
        # Initializing the gradients
        d_weights_XH = np.zeros_like(self.weights_XH)
        d_weights_HH = np.zeros_like(self.weights_HH)
        d_bias_h = np.zeros_like(self.bias_h)
        d_weights_HO = np.zeros_like(self.weights_HO)
        d_bias_O = np.zeros_like(self.bias_O)
        
        # Initialize the gradients for the hidden state
        d_h_next = np.zeros((1, self.hidden_lr_size))
        
        # Loop through the sequence in reverse
        for t in reversed(range(x.shape[0])):
            # Geting the hidden state at time t
            h = self.hidden_states[t]
            # Calculating the output error
            output_error = self.outputs[-1] - y
            # Calculating the gradients for the outputing layer
            d_weights_HO += np.dot(h.T, output_error)
            d_bias_O += output_error
            # Calculating the gradients for the hidden state
            d_h = np.dot(output_error, self.weights_HO.T) + d_h_next
            d_h_raw = d_h * (1 - h ** 2)
            # Calculating the gradients for the input layer-to-hidden layer
            d_weights_XH += np.dot(x[t].reshape((input_lr_size, 1)), d_h_raw)
            # Calculating the gradients for the hidden-to-hidden layer
            if t > 0:
                h_prev = self.hidden_states[t - 1]
                d_weights_HH += np.dot(h_prev.T, d_h_raw)
            d_bias_h += d_h_raw
            # Updating the next hidden state gradient
            d_h_next = np.dot(d_h_raw, self.weights_HH.T)
        
        # Updating the parameters
        self.weights_XH -= self.learningRate * d_weights_XH
        self.weights_HH -= self.learningRate * d_weights_HH
        self.bias_h -= self.learningRate * d_bias_h
        self.weights_HO -= self.learningRate * d_weights_HO
        self.bias_O -= self.learningRate * d_bias_O
        
    def train(self, trainPart_X, trainPart_Y, epochs=100, batchSize=32):
        n_samples = trainPart_X.shape[0]
        for epoch in range(epochs):
            for i in range(0, n_samples, batchSize):
                batchPart_X = trainPart_X[i:i + batchSize]
                batchPart_Y = trainPart_Y[i:i + batchSize]
                for j in range(len(batchPart_X)):
                    x = batchPart_X[j]
                    y = batchPart_Y[j]
                    # Forward pass
                    self.forward(x)
                    # Backward pass
                    self.backward(x, y)
            
            # Calculating loss after each epoch
            loss = self.calculate_loss(trainPart_X, trainPart_Y)
            print(f"Epoch {epoch + 1}/{epochs}, Encountered Loss: {loss:.5f}")
    
    def calculate_loss(self, trainPart_X, trainPart_Y):
        # Calculating loss for the entire nyse_dataset
        total_loss = 0.0

        # Iterating through each sample in the training set
        for i in range(trainPart_X.shape[0]):
            x = trainPart_X[i]
            y = trainPart_Y[i]
            # Performing forward pass to get the prediction
            prediction = self.forward(x)
            # Calculating squared error (loss) for the current sample
            sample_loss = (prediction - y) ** 2
            # Accumulating the loss
            total_loss += sample_loss

        # Calculating average loss
        mean_loss = total_loss / trainPart_X.shape[0]

        # Converting mean loss to a scalar value if it is an ndarray
        if isinstance(mean_loss, np.ndarray):
            mean_loss = mean_loss.item()

        return mean_loss

    def predict(self, testPart_X):
        predictions = []
        for i in range(testPart_X.shape[0]):
            x = testPart_X[i]
            prediction = self.forward(x)
            predictions.append(prediction)
        return np.array(predictions)

# Initializing the RNN model
input_lr_size = trainPart_X.shape[2]  # Number of features in the input
hidden_lr_size = 50
output_lr_size = 1
learningRate = 0.01

rnn = RNN(input_lr_size, hidden_lr_size, output_lr_size, learningRate)

# Training the model
rnn.train(trainPart_X, trainPart_Y, epochs=50, batchSize=32)

# Predicting the test part
predicted_data = rnn.predict(testPart_X)

# Ensuring predicted_data is in the shape of (n_samples, 1)
predicted_data = predicted_data[:, -1, 0]

# Calculating the mean absolute error(MAE)
mae = mean_absolute_error(testPart_Y, predicted_data)
print(f"Mean Absolute Error (MAE): {mae:.5f}")

# Ploting the results
plt.plot(range(len(predicted_data)), predicted_data, color='orange', label='Predicted Stock Price')
plt.plot(range(len(testPart_Y)), testPart_Y, color='black', label='Actual Stock Price')
plt.title("Predicted Stock Price vs Actual Stock Price")
plt.xlabel('Time')
plt.ylabel('AAPL Stock Price')
plt.legend()
plt.show()
