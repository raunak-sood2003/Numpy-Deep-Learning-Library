import numpy as np
from npdl.nn.module import Layer

class Linear(Layer):
    '''
    Linear Activation (identity function)
    
    Args: None
    
    Equation: 
        f(x) = x 
    
    Notes: no change from input to output; just for readability
    
    '''
    def __init__(self):
        super().__init__("linear_act", False)
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = self.inputs
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class ReLU(Layer):
    '''
    Rectified Linear Unit
    
    Args: None
    
    Equation: 
        f(x) = x for x > 0
        f(x) = 0 for x <= 0
    
    Notes: 
        Source: https://arxiv.org/abs/1803.08375
    
    '''
    def __init__(self):
        super().__init__("relu_act", False)
        
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class LeakyReLU(Layer):
    '''
    Leaky Rectified Linear Unit
    
    Args:
        alpha - scale for negative values
    
    Equation: 
        f(x) = x for x > 0
        f(x) = alpha * x for x <= 0
    
    Notes: 
        Also known as Parametric ReLU (PReLU)
        Source: https://arxiv.org/abs/1502.01852
    
    '''
    def __init__(self, alpha = 0.01):
        super().__init__("lrelu_act", False)
        self.alpha = alpha
        
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, inputs * self.alpha)
    
    def backward(self, dvalues):
        self.dinputs = np.where(self.inputs <= 0, self.alpha * dvalues, dvalues)

class ELU(Layer):
    '''
    Exponential Linear Unit
    
    Args:
        alpha - scale for negative values
    
    Equation: 
        f(x) = x for x >= 0
        f(x) = alpha * (exp(x) - 1) for x < 0
    
    Notes: 
        Source: https://arxiv.org/abs/1511.07289
    
    '''
    def __init__(self, alpha = 1.0):
        super().__init__("elu_act", False)
        self.alpha = alpha
        
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * (np.exp(inputs) - 1))
    
    def backward(self, dvalues):
        self.dinputs = np.where(self.inputs <= 0, self.alpha * np.exp(self.inputs) * dvalues, dvalues)
    
class Sigmoid(Layer):
    '''
    Sigmoid Activation
    
    Args: None
    
    Equation: 
        f(x) = 1 / (1 + exp(-x))
    
    Notes: Scales output between 0 and 1 (used in binary classification)
    
    '''
    def __init__(self):
        super().__init__("sigmoid_act", False)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = self.sigmoid(inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)
        
class Tanh(Layer):
    '''
    Hyperbolic Tangent Activation
    
    Args: None
    
    Equation: 
        f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Notes: Scales output between -1 and 1
    
    '''
    def __init__(self):
        super().__init__("tanh_act", False)
        
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.tanh(inputs)
    
    def backward(self, dvalues):
        self.dinputs = (1 - ((self.output) ** 2)) * dvalues
    
class Softmax(Layer):
    '''
    Softmax Activation
    
    Args: None
    
    Equation: 
        f(x_i) = exp(x_i) / sum(exp(x_j)) | j = 1 to K
    
    Notes:
         - Input x_i is a vector and K is the number of classes
         - This implementation subtracts the maximum value from 
           the input to prevent 'weight explosion' due to the 
           nature of the exponential function.
         - Source: https://dl.acm.org/doi/10.5555/2969830.2969856
    
    '''
    def __init__(self):
        super().__init__("softmax_act", False)
    
    def forward(self, inputs, training):
        self.inputs = inputs
        inputs_max = np.max(inputs, axis = 1, keepdims = True) 
        exp_values = np.exp(inputs - inputs_max) #Prevents weight explosion
        self.output = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index, (s_output, s_dvalues) in enumerate(zip(self.output, dvalues)):
            s_output = s_output.reshape(-1, 1)
            jacobian = np.diagflat(s_output) - np.dot(s_output, s_output.T)
            self.dinputs[index] = np.dot(jacobian, s_dvalues)

class Softplus(Layer):
    '''
    Softplus Activation
    
    Args: None
    
    Equation: 
        f(x) = log (1 + exp(x))
    
    Notes: 
        - Smooth (differentiable) version of ReLU
        - Source: https://ieeexplore.ieee.org/document/7280459
    
    '''
    def __init__(self):
        super().__init__("softplus_act", False)
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.log(1 + np.exp(inputs))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, dvalues):
        self.dinputs = self.sigmoid(self.inputs) * dvalues