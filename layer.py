import numpy as np
from typing import Callable, Optional

def sigmoid(x: np.ndarray) -> np.ndarray:
    """This function computes the sigmoid activation.
    formula: \frac{1}{1 + e^{-x}}
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)

def relu(x: np.ndarray) -> np.ndarray:
    """This function computes the ReLU activation.
    formula: max(0, x)
    """
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def tanh(x: np.ndarray) -> np.ndarray:
    """This function computes the tanh activation.
    formula: \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}
    """
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2

def linear(x: np.ndarray) -> np.ndarray:
    """This function computes the linear activation.
    formula: x
    """
    return x

def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)

def softmax(x: np.ndarray) -> np.ndarray:
    """This function computes the softmax activation.
    formula: \frac{e^{x_i}}{\sum_{j} e^{x_j}}
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_derivative(x: np.ndarray) -> np.ndarray:
    s = softmax(x)
    return s * (1 - s)

ACTIVATIONS: dict[str, dict[str, Callable]] = {
    "sigmoid": {"function": sigmoid, "derivative": sigmoid_derivative},
    "relu": {"function": relu, "derivative": relu_derivative},
    "tanh": {"function": tanh, "derivative": tanh_derivative},
    "linear": {"function": linear, "derivative": linear_derivative},
    "softmax": {"function": softmax, "derivative": softmax_derivative},
}