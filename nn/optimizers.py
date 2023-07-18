import numpy as np

'''
Optimizers

Common Args
    lr - learning rate
    lr_decay - parameter between 0 and 1 indicating how much the learning 
    rate should decrease based on the number of iterations of gradient descent

Notes:
    - All algorithms are variants of stochastic gradient descent
        * W_new = W_old - lr * (dL / dW)
        * b_new = b_old - lr * (dL / db)
    
'''

class SGD:
    '''
    Optimizer implementing Stochastic Gradient Descent
    
    Args:
        momentum - exponetially weighted average parameter to update weights
            * dW := dW * momentum + dW_new * (1 - momentum)
            * W := W - lr * dW
    
    Notes:
        - Momentum helps with the local minimum issue and should be used
        - Vanilla SGD will likely me insufficient for many cases, but is 
          a necessary implementation for all later optimizers
    
    '''
    def __init__(self, lr = 1e-3, lr_decay = 0.0, momentum = 0.0):
        self.lr = lr
        self.curr_lr = lr
        self.lr_decay = lr_decay
        self.iterations = 0
        self.momentum = momentum
    
    def pre_update_params(self):
        '''
        Helper Function Before Optimizer Runs
            * Sets up learning rate decay
        
        :returns VOID
        '''
        self.curr_lr = self.lr / (1 + self.iterations * self.lr_decay)
    
    def update_params(self, layer):
        '''
        Parameter Update Function
        
        :param layer: layer object from npdl.nn.layers
        
        :returns VOID
        '''
        if self.momentum != 0:
            if not hasattr(layer, 'weight_momentum'):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentum - self.curr_lr * layer.dweights
            layer.weight_momentum = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentum - self.curr_lr * layer.dbiases
            layer.bias_momentum = bias_updates
        
        else:
            weight_updates = -self.curr_lr * layer.dweights
            bias_updates = -self.curr_lr * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        '''
        Helper Function After Optimizer Runs
            * Increases the number of iterations for LR decay
            
        :returns VOID
        '''
        self.iterations += 1

class AdaGrad:
    ''' 
    Optimizer implementing the AdaGrad algorithm
    
    Args:
        epsilon - small constant value preventing division by zero when 
        dividing by the weight/bias cache
        
    Notes:
        - Source: https://jmlr.org/papers/v12/duchi11a.html
        - AdaGrad is an adaptive learning rate algorihtm; it updates parameters inversely
          proportional to the number of updates to the parameter
    
    '''
    def __init__(self, lr = 1e-3, lr_decay = 0.0, epsilon = 1e-7):
        self.lr = lr
        self.curr_lr = lr
        self.lr_decay = lr_decay
        self.iterations = 0
        self.epsilon = epsilon
    
    def pre_update_params(self):
        '''
        Helper Function Before Optimizer Runs
            * Sets up learning rate decay
        
        :returns VOID
        '''
        self.curr_lr = self.lr / (1 + self.iterations * self.lr_decay)
    
    def update_params(self, layer):
        '''
        Parameter Update Function
        
        :param layer: layer object from npdl.nn.layers
        
        :returns VOID
        '''
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        
        layer.weights -= self.curr_lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases -= self.curr_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        '''
        Helper Function After Optimizer Runs
            * Increases the number of iterations for LR decay
            
        :returns VOID
        '''
        self.iterations += 1

class RMSProp:
    '''
    Optimizer implementing the Root Mean Square Propagation algorithm
    
    Args:
        epsilon - small constant value preventing division by zero when
        dividing by the weight/bias cache
        rho - parameter for exponentially weighted average of the weight/bias cache
    
    Notes
        - Source: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        - RMSProp combines momentum and AdaGrad by keeping a weight cache for adaptive updates
          and using exponentially weighted averages.
    
    '''
    def __init__(self, lr = 1e-3, lr_decay = 0.0, epsilon = 1e-7, rho = 0.9):
        self.lr = lr
        self.curr_lr = lr
        self.lr_decay = lr_decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    def pre_update_params(self):
        '''
        Helper Function Before Optimizer Runs
            * Sets up learning rate decay
        
        :returns VOID
        '''
        self.curr_lr = self.lr / (1 + self.iterations * self.lr_decay)
    
    def update_params(self, layer):
        '''
        Parameter Update Function
        
        :param layer: layer object from npdl.nn.layers
        
        :returns VOID
        '''
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
        
        layer.weights -= self.curr_lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases -= self.curr_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        '''
        Helper Function After Optimizer Runs
            * Increases the number of iterations for LR decay
            
        :returns VOID
        '''
        self.iterations += 1
        
class Adam:
    '''
    Optimizer implementing the Adaptive Moment Estimation (Adam) algorithm
    
    Args:
        epsilon - small constant value preventing division by zero when
        dividing by the weight/bias cache
        beta1 - parameter for exponential weighted average of the gradients
        beta2 - parameter for exponential weighted average of the weight/bias cache
        
    Notes:
        - Source: https://arxiv.org/abs/1412.6980
    
    '''
    def __init__(self, lr = 1e-3, lr_decay = 0.0, epsilon = 1e-7, beta1 = 0.9, beta2 = 0.999):
        self.lr = lr
        self.curr_lr = lr
        self.lr_decay = lr_decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    
    def pre_update_params(self):
        '''
        Helper Function Before Optimizer Runs
            * Sets up learning rate decay
        
        :returns VOID
        '''
        self.curr_lr = self.lr / (1 + self.iterations * self.lr_decay)
    
    def update_params(self, layer):
        '''
        Parameter Update Function
        
        :param layer: layer object from npdl.nn.layers
        
        :returns VOID
        '''
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        
        layer.weight_momentum = self.beta1 * layer.weight_momentum + (1 - self.beta1) * layer.dweights
        layer.bias_momentum = self.beta1 * layer.bias_momentum + (1 - self.beta1) * layer.dbiases
        weight_momentum_normalized = layer.weight_momentum / (1 - self.beta1 ** (self.iterations + 1))
        bias_momentum_normalized = layer.bias_momentum / (1 - self.beta1 ** (self.iterations + 1))
        
        
        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights ** 2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases ** 2
        weight_cache_normalized = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_normalized = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))
        
        layer.weights -= self.curr_lr * weight_momentum_normalized / (np.sqrt(weight_cache_normalized) + self.epsilon)
        layer.biases -= self.curr_lr * bias_momentum_normalized / (np.sqrt(bias_cache_normalized) + self.epsilon) 
    
    def post_update_params(self):
        '''
        Helper Function After Optimizer Runs
            * Increases the number of iterations for LR decay
            
        :returns VOID
        '''
        self.iterations += 1
