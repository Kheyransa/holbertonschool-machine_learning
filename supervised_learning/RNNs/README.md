# RNNs

## Description

This project covers the implementation of Recurrent Neural Networks (RNNs)
from scratch using `numpy`, without relying on deep learning frameworks
such as TensorFlow or Keras. The goal is to understand the internal
mechanics of RNN cells and their variants (GRU, LSTM, bidirectional
RNNs) by building forward propagation step by step.

## Learning Objectives

At the end of this project, you should be able to explain:

- What is a recurrent neural network (RNN)
- The difference between a simple RNN cell, GRU cell, and LSTM cell
- What is a bidirectional RNN
- How to represent an RNN cell with `numpy`
- How to perform forward propagation through time (an unrolled RNN)
- How gates work inside GRU and LSTM cells

## Requirements

- Ubuntu 20.04 LTS
- Python 3.9
- numpy (only)
- All files must follow the pycodestyle style guide (version 2.x)
- All files must be executable
- All modules, classes, and functions must have documentation

## Files

| File               | Description                                              |
|--------------------|-----------------------------------------------------------|
| `0-rnn_cell.py`    | `RNNCell` class: represents a cell of a simple RNN         |

### 0. RNN Cell

`RNNCell` represents a single cell of a simple RNN.

**Class constructor:** `def __init__(self, i, h, o)`

- `i` is the dimensionality of the data
- `h` is the dimensionality of the hidden state
- `o` is the dimensionality of the outputs
- Creates the public instance attributes `Wh`, `Wy`, `bh`, `by` that
  represent the weights and biases of the cell:
  - `Wh` and `bh` are for the concatenated hidden state and input data
  - `Wy` and `by` are for the output
  - The weights are initialized using a random normal distribution
  - The biases are initialized as zeros

**Public instance method:** `def forward(self, h_prev, x_t)`

Performs forward propagation for one time step.

- `x_t` is a `numpy.ndarray` of shape `(m, i)` that contains the data
  input for the cell (`m` is the batch size)
- `h_prev` is a `numpy.ndarray` of shape `(m, h)` containing the
  previous hidden state
- The output of the cell uses a softmax activation function
- Returns: `h_next, y`
  - `h_next` is the next hidden state
  - `y` is the output of the cell

## Author

Holberton School - Machine Learning Specialization
