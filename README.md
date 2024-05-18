# RNN Stock Price Prediction
This repository contains code for training and testing a Recurrent Neural Network (RNN) model from scratch to predict stock prices using historical data from the New York Stock Exchange (NYSE). The model is implemented using Python and utilizes libraries such as NumPy, Pandas, Matplotlib, and scikit-learn.

## Introduction
The code in this repository demonstrates how to build an RNN model to predict stock prices based on historical data. It includes functions for data preprocessing, model training, testing, and evaluation. The RNN architecture used in this implementation includes a single hidden layer.

## Installation
To run the code, you'll need Python 3.x along with the following libraries:

- NumPy
- Pandas
- Matplotlib
- scikit-learn

You can install these dependencies using pip:
```bash
pip install numpy pandas matplotlib scikit-learn
```
## Usage
### 1. Clone this repository:
```bash
git clone https://github.com/prathameshk03/rnn-stock-price-prediction.git
```
### 2. Navigate to the project directory:
```bash
cd rnn-stock-price-prediction
```
### 3. Run the main script:
```bash
python rnn_stock_prediction.py
```
This will train the RNN model and display the predicted stock prices along with the actual prices.

## Dataset
The dataset used for training and testing the model is sourced from the New York Stock Exchange (NYSE). It includes historical stock price data for various companies, including opening price, closing price, high, low, and volume.

## Model Architecture
The RNN model architecture consists of:

- Input Layer: Accepts sequences of historical stock prices.
- Hidden Layer: Contains recurrent units responsible for capturing temporal dependencies.
- Output Layer: Predicts the future stock price based on the input sequence.
The model utilizes the tanh activation function and Mean Squared Error (MSE) loss during training.

## Results
After training the model, the predicted stock prices are compared with the actual prices using Mean Absolute Error (MAE) as the evaluation metric. Additionally, a plot is generated to visualize the predicted versus actual stock prices.
