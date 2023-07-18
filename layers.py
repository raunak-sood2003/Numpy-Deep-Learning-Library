import numpy as np
from npdl.nn.module import Layer
from npdl.nn.layer_tools import get_im2col_indices, im2col_indices, col2im_indices

'''
********************************************** Neural Network Layers **********************************************

'''

class Linear(Layer):
    '''
    Fully Connected (Dense) Layer
    
    Args:
        n_inputs - number of neurons in the previous layer (or input layer if first dense layer)
        n_neurons - number of neurons in this layer
        init - type of weight intitialization
            * xavier uniform initializaton: https://proceedings.mlr.press/v9/glorot10a.html
            * xavier normal initializaton: https://proceedings.mlr.press/v9/glorot10a.html
            * he initializaton: https://arxiv.org/abs/1502.01852
        weight_reg_l1 - L1 regularization parameter for weights (between 0 and 1)
        bias_reg_l1 - L1 regularization parameter for biases (between 0 and 1)
        weight_reg_l2 - L2 regularization parameter for weights (between 0 and 1)
        bias_reg_l2 - L2 regularization parameter for biases (between 0 and 1)
        
    '''
    def __init__(self, n_inputs, n_neurons, init = 'xavier_normal', weight_reg_l1 = 0.0, \
                     bias_reg_l1 = 0.0, weight_reg_l2 = 0.0, bias_reg_l2 = 0.0):
        super().__init__("linear", True)
        self.weights = self.init_weights(n_inputs, n_neurons, init)
        self.biases = np.zeros([1, n_neurons])
        
        self.weight_reg_l1 = weight_reg_l1
        self.bias_reg_l1 = bias_reg_l1
        self.weight_reg_l2 = weight_reg_l2
        self.bias_reg_l2 = bias_reg_l2 
    
    def init_weights(self, n_rows, n_cols, init):
        '''
        Weight Initialization Scheme for Linear Layer
        
        :param n_rows: (int) number of neurons in the previous layer
        :param n_cols: (int) number of neurons in the current layer
        :param init: (str) initialization scheme
        
        :returns numpy.ndarray of shape (n_rows, n_cols), a weight matrix 
            intiialized according to init
        
        '''
        if init == 'xavier_normal':
            std = np.sqrt(2.0 / (n_rows + n_cols))
            weights = np.random.normal(loc = 0.0, scale = std, size = (n_rows, n_cols))
        
        elif init == 'xavier_uniform':
            bound = np.sqrt(6.0 / (n_rows + n_cols))
            weights = np.random.uniform(low = -bound, high = bound, size = (n_rows, n_cols))
        
        elif init == 'he_normal':
            std = np.sqrt(2.0 / n_rows)
            weights = np.random.normal(loc = 0.0, scale = std, size = (n_rows, n_cols)) 
        
        elif init == 'he_uniform':
            bound = np.sqrt(6.0 / n_rows)
            weights = np.random.uniform(low = -bound, high = bound, size = (n_rows, n_cols)) 
        
        else:
            weights = 0.1 * np.random.randn(n_rows, n_cols)

        return weights
    
    def forward(self, inputs, training):
        '''
        Forward Pass of Linear Layer
        
        :param inputs: (numpy.ndarray) input array of size (N, k) where N is 
             the batch size and k is the number of input dimensions
        
        :param training: (bool) indicates whether the layer is currently training or not
        
        :returns VOID
        
        '''
        self.inputs = inputs
        self.output = np.matmul(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        '''
        Backward Pass of Linear Layer
        
        :param dvalues: (numpy.ndarray) derivative of the loss function with respect 
            to the output; size: (N, k') where N is the batch size and k' is 
            the number of output dimensions
        
        :returns VOID
        
        '''
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        
        dl1_b = np.ones_like(self.biases)
        dl1_w = np.ones_like(self.weights)
        dl1_b[self.biases < 0] = -1
        dl1_w[self.weights < 0] = -1
        
        self.dbiases += self.bias_reg_l1 * dl1_b
        self.dweights += self.weight_reg_l1 * dl1_w
        self.dbiases += 2 * self.bias_reg_l2 * self.biases
        self.dweights += 2 * self.weight_reg_l2 * self.weights
        
        self.dinputs = np.dot(dvalues, self.weights.T)

class Conv2D(Layer):
    '''
    2-Dimensional Convolutional Layer
    
    Args:
        in_channels - number of channels from the previous layer
        out_channels - number of channels in the current layer
        kernel_size - size of the convolutional kernel (height, width)
        stride - stride length of kernel (strideX, strideY)
        padding - zero padding applied to input (padX, padY)
        init - type of weight initialization
            * xavier uniform initializaton: https://proceedings.mlr.press/v9/glorot10a.html
            * xavier normal initializaton: https://proceedings.mlr.press/v9/glorot10a.html
            * he initializaton: https://arxiv.org/abs/1502.01852
        weight_reg_l1 - L1 weight regularization parameter
        bias_reg_l1 - L1 bias regularization parameter
        weight_reg_l2 - L2 weight regularization parameter
        bias_reg_l2 - L2 bias regularization parameter
    
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride = (1, 1), padding = (0, 0), \
                     init = 'xavier_normal', weight_reg_l1 = 0.0, bias_reg_l1 = 0.0, \
                         weight_reg_l2 = 0.0, bias_reg_l2 = 0.0):
        
        super().__init__("Conv2D", True)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weights = self.init_weights((out_channels, in_channels, kernel_size[0], kernel_size[1]), init)
        self.biases = np.zeros([out_channels, 1])
        
        self.weight_reg_l1 = weight_reg_l1
        self.bias_reg_l1 = bias_reg_l1
        self.weight_reg_l2 = weight_reg_l2
        self.bias_reg_l2 = bias_reg_l2 
    
    def init_weights(self, weight_shape, init):
        '''
        Weight Initialization Scheme for Convolutional Layer
        
        :param weight_shape: (int tuple) shape of weight tensor; shape: (c_out, c_in, H, W) 
         where c_out is the number of out-channels, c_in is the number of input channels 
         and (H, W) is the shape of the individual kernel 
        
        :param init: (str) initialization scheme
        
        :returns numpy.ndarray of shape weight_shape, a weight matrix 
            intiialized according to init
        
        '''
        out_channels = weight_shape[0]
        in_channels = weight_shape[1]
        kernel_height = weight_shape[2]
        kernel_width = weight_shape[3]

        n_in = in_channels * kernel_height * kernel_width
        n_out = in_channels * kernel_height * kernel_width

        if init == 'xavier_normal':
            std = np.sqrt(2.0 / (n_in + n_out))
            weights = np.random.normal(loc = 0.0, scale = std, size = weight_shape)

        elif init == 'xavier_uniform':
            bound = np.sqrt(6.0 / (n_in + n_out))
            weights = np.random.uniform(low = -bound, high = bound, size = weight_shape)

        elif init == 'he_normal':
            std = np.sqrt(2.0 / n_in)
            weights = np.random.normal(loc = 0.0, scale = std, size = weight_shape)
        
        elif init == 'he_uniform':
            bound = np.sqrt(6.0 / n_in)
            weights = np.random.uniform(low = -bound, high = bound, size = weight_shape)
        
        else:
            weights = 0.1 * np.random.randn(weight_shape)

        return weights
    
    def forward(self, inputs, training):
        '''
        Forward Pass of Convolutional Layer
        
        :param inputs: (numpy.ndarray) input array of size (N, C, H, W) where N is 
             the batch size, C is the number of channels and (H, W) is the height and width
        
        :param training: (bool) indicates whether the layer is currently training or not
        
        :returns VOID
        
        '''
        self.inputs = inputs
        
        n_in, c_in, h_in, w_in = inputs.shape
        h_kernel, w_kernel = self.kernel_size
        h_pad, w_pad = self.padding
        h_stride, w_stride = self.stride
        
        self.inputs_col = im2col_indices(inputs, h_kernel, w_kernel, stride = self.stride, \
                                              padding = self.padding)
        
        weights_row = self.weights.reshape(self.out_channels, -1)
        
        h_out = int(1 + (h_in - h_kernel + 2 * h_pad) // h_stride)
        w_out = int(1 + (w_in - w_kernel + 2 * w_pad) // w_stride)
        
        self.output = weights_row @ self.inputs_col + self.biases
        self.output = self.output.reshape(self.out_channels, h_out, w_out, n_in)
        self.output = self.output.transpose(3, 0, 1, 2)

    def backward(self, dvalues):
        '''
        Backward Pass of Convolutional Layer
        
        :param dvalues: (numpy.ndarray) derivative of the loss function with respect 
            to the output; size: (N, C', H', W') where N is the batch size, C' is the 
            output channels, and (H', W') is the output height and width
        
        :returns VOID
        
        '''
        dvalues_flat = dvalues.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        self.dweights = dvalues_flat @ self.inputs_col.T
        self.dweights = self.dweights.reshape(self.weights.shape)

        self.dbiases = np.sum(dvalues, axis=(0, 2, 3)).reshape(self.out_channels, -1)

        weights_flat = self.weights.reshape(self.out_channels, -1)

        dinputs_col = weights_flat.T @ dvalues_flat
        
        self.dinputs = col2im_indices(dinputs_col, self.inputs.shape, self.kernel_size[0], \
                                           self.kernel_size[1], self.padding, self.stride)

class Flatten(Layer):
    '''
    Flattening Layer
    
    Args: None
    
    Notes:
        - Converts high dimensional vectors to one 
          dimensional vector for Linear layers
    
    '''
    def __init__(self):
        super().__init__("flatten", False)
    
    def forward(self, inputs, training):
        '''
        Forward Pass of Flatten Layer
        
        :param inputs: (numpy.ndarray) input array of size (N, C, H, W) where N is 
             the batch size, C is the number of channels and (H, W) is the height and width
        
        :param training: (bool) indicates whether the layer is currently training or not
        
        :returns VOID
        
        '''
        self.inputs = inputs
        
        batch_size = inputs.shape[0]
        in_channels = inputs.shape[1]
        in_height = inputs.shape[2]
        in_width = inputs.shape[3]
        
        self.output = inputs.reshape(batch_size, in_channels * in_height * in_width)
    
    def backward(self, dvalues):
        '''
        Backward Pass of Flatten Layer
        
        :param dvalues: (numpy.ndarray) derivative of the loss function with respect 
            to the output; size: (N, C', H', W') where N is the batch size, C' is the 
            output channels, and (H', W') is the output height and width
        
        :returns VOID
        
        '''
        self.dinputs = dvalues.reshape(self.inputs.shape)

class MaxPool2D(Layer):
    '''
    2-Dimensional Max-Pooling Layer
    
    Args:
        kernel_size - size of the pooling kernel (height, width)
        stride - stride length of kernel (strideX, strideY)
        padding - zero padding applied to input (padX, padY)
    
    '''
    def __init__(self, pool_size, stride = (1, 1), padding = (0, 0)):
        super().__init__("MaxPool2D", False)
        
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, inputs, training):
        '''
        Forward Pass of Max-Pooling Layer
        
        :param inputs: (numpy.ndarray) input array of size (N, C, H, W) where N is 
             the batch size, C is the number of channels and (H, W) is the height and width
        
        :param training: (bool) indicates whether the layer is currently training or not
        
        :returns VOID
        
        '''
        self.inputs = inputs
        self.n_in, self.c_in, self.h_in, self.w_in = inputs.shape
        
        self.h_out = int(1 + (self.h_in - self.pool_size[0] + 2 * self.padding[0]) // self.stride[0])
        self.w_out = int(1 + (self.w_in - self.pool_size[1] + 2 * self.padding[1]) // self.stride[1])
        
        X_reshaped = inputs.reshape(self.n_in * self.c_in, 1, self.h_in, self.w_in)

        self.X_col = im2col_indices(X_reshaped, self.pool_size[0], self.pool_size[1], \
                                         padding = (0, 0), stride = self.stride)
        
        self.max_indexes = np.argmax(self.X_col, axis=0)
        
        self.output = self.X_col[self.max_indexes, range(self.max_indexes.size)]
        self.output = self.output.reshape(self.h_out, self.w_out, self.n_in, self.c_in).transpose(2, 3, 0, 1)
    
    def backward(self, dvalues):
        '''
        Backward Pass of Max-Pooling Layer
        
        :param dvalues: (numpy.ndarray) derivative of the loss function with respect 
            to the output; size: (N, C', H', W') where N is the batch size, C' is the 
            output channels, and (H', W') is the output height and width
        
        :returns VOID
        
        '''
        dX_col = np.zeros_like(self.X_col)
        dout_flat = dvalues.transpose(2, 3, 0, 1).ravel()
        dX_col[self.max_indexes, range(self.max_indexes.size)] = dout_flat

        shape = (self.n_in * self.c_in, 1, self.h_in, self.w_in)
        
        self.dinputs = col2im_indices(dX_col, shape, self.pool_size[0], self.pool_size[1], \
                                 padding = (0, 0), stride = self.stride)
        
        self.dinputs = self.dinputs.reshape(self.inputs.shape)        

class Dropout(Layer):
    '''
    Dropout Layer
    
    Args
        dropout_rate - parameter (between 0 and 1) indicating the probability of a given 
        neuron in the layer to be deactivated during training
        
    Notes:
        - Source: https://jmlr.org/papers/v15/srivastava14a.html
        - This layer randomly drops certain neurons from training according to the above paper
        - Must only be active during training and NOT during inference 
          Can be deactivated by setting 'training' to False during inference
        
    '''
    def __init__(self, dropout_rate):
        super().__init__("dropout", False)
        self.dropout_rate = dropout_rate
            
    def forward(self, inputs, training):
        '''
        Forward Pass of Dropout Layer
        
        :param inputs: (numpy.ndarray) input array of size (N, k) for Linear layers or (N, C, H, W) where N is 
             the batch size, k is the number of neurons in the previous layer, C is the number of channels and 
             (H, W) is the height and width
        
        :param training: (bool) indicates whether the layer is currently training or not
        
        :returns VOID
        
        '''
        if not training:
            self.output = inputs.copy()
            return
        self.inputs = inputs
        self.mask = np.random.binomial(1, 1 - self.dropout_rate, size = inputs.shape) / (1 - self.dropout_rate)
        self.output = inputs * self.mask
    
    def backward(self, dvalues):
        '''
        Backward Pass of Dropout Layer
        
        :param dvalues: (numpy.ndarray) derivative of the loss function with respect 
            to the output; size: (N, k') for Linear layers or (N, C', H', W') for Convolutional layers 
            where N is the batch size, k' is the number of neurons in the next layer, C' is the 
            output channels, and (H', W') is the output height and width
        
        :returns VOID
        
        '''
        self.dinputs = dvalues * self.mask

class BatchNorm1D(Layer):
    '''
    1-Dimensional Batch Normalization
    
    Args:
        n_inputs - number of neurons in previous 'Linear' layer
        epsilon - small constant used to prevent division by zero during normalization
        momentum - parameter for exponentially weighted average of mean and variance for inference phase
    
    Notes:
         - Source: https://arxiv.org/abs/1502.03167
         - Training: current batch mean and variance is used during training
         - Testing: we keep track of moving averages of mean and variance for inputs
           and use this value during testing (see above paper for more details)
    
    '''
    def __init__(self, n_inputs, epsilon = 1e-7, momentum = 0.99):
        super().__init__("batch_norm", True)
        self.weights = np.ones([1, n_inputs])
        self.biases = np.zeros([1, n_inputs])
        self.epsilon = epsilon
        self.momentum = momentum
        
    def forward(self, inputs, training):
        '''
        Forward Pass of BatchNorm1D Layer
        
        :param inputs: (numpy.ndarray) input array of size (N, k) where N is 
             the batch size and k is the number of input dimensions
        
        :param training: (bool) indicates whether the layer is currently training or not
        
        :returns VOID
        
        '''
        self.mean = np.mean(inputs, axis = 0, keepdims = True)
        self.var = np.var(inputs, axis = 0, keepdims = True)

        if not hasattr(self, 'mean_moving'):
            self.mean_moving = np.zeros_like(self.mean)
            self.var_moving = np.zeros_like(self.var)

        self.mean_moving = self.momentum * self.mean_moving + (1 - self.momentum) * self.mean
        self.var_moving = self.momentum * self.var_moving + (1 - self.momentum) * self.var
        
        if not training: # Inference Mode for Batch Norm; use moving average instead of batch averages
            self.inputs_norm = (inputs - self.mean_moving) / np.sqrt(self.var_moving + self.epsilon)
            self.output = self.inputs_norm * self.weights
            self.output += self.biases
            return
        
        self.inputs_norm = (inputs - self.mean) / np.sqrt(self.var + self.epsilon)
        self.output = self.inputs_norm * self.weights
        self.output += self.biases
        
    
    def backward(self, dvalues):
        '''
        Backward Pass of Linear Layer
        
        :param dvalues: (numpy.ndarray) derivative of the loss function with respect 
            to the output; size: (N, k') where N is the batch size and k' is 
            the number of output dimensions
        
        :returns VOID
        
        '''
        self.dweights = np.sum(dvalues * self.inputs_norm, axis = 0).reshape([1 ,-1])
        self.dbiases = np.sum(dvalues, axis = 0).reshape([1 ,-1])
        
        N = dvalues.shape[0]
        
        ones = np.ones([N, 1])
        self.dinputs = self.weights * (1 / (N * np.sqrt(self.var + self.epsilon))) * \
                (-self.dweights * self.inputs_norm + N * dvalues - np.dot(ones, self.dbiases))

class BatchNorm2D(Layer):
    '''
    2-Dimensional Batch Normalization
    
    Args:
        n_channels - number of channels in previous 'Convolutional' layer
        epsilon - small constant used to prevent division by zero during normalization
        momentum - parameter for exponentially weighted average of mean and variance for inference phase
    
    Notes:
         - Source: https://arxiv.org/abs/1502.03167
         - Training: current batch mean and variance is used during training
         - Testing: we keep track of moving averages of mean and variance for inputs
           and use this value during testing (see above paper for more details)
    
    '''
    def __init__(self, n_channels, epsilon = 1e-7, momentum = 0.99):
        super().__init__("BatchNorm2D", True)
    
        self.n_channels = n_channels
        self.weights = np.ones([n_channels,])
        self.biases = np.zeros([n_channels,])
        self.epsilon = epsilon
        self.momentum = momentum
    
    def forward(self, inputs, training):
        '''
        Forward Pass of BatchNorm2D Layer
        
        :param inputs: (numpy.ndarray) input array of size (N, C, H, W) where N is 
             the batch size, C is the number of channels and (H, W) is the height and width
        
        :param training: (bool) indicates whether the layer is currently training or not
        
        :returns VOID
        
        '''
        self.inputs = inputs
        n_in, c_in = inputs.shape[0], inputs.shape[1]
        assert c_in == self.n_channels
        
        self.mean = np.mean(inputs, axis = (0, 2, 3)).reshape(1, c_in, 1, 1)
        self.var = np.var(inputs, axis = (0, 2, 3)).reshape(1, c_in, 1, 1)
        
        if not hasattr(self, 'mean_moving'):
            self.mean_moving = np.zeros_like(self.mean)
            self.var_moving = np.zeros_like(self.var)

        self.mean_moving = self.momentum * self.mean_moving + (1 - self.momentum) * self.mean
        self.var_moving = self.momentum * self.var_moving + (1 - self.momentum) * self.var
        
        if not training:
            self.inputs_norm = (inputs - self.mean_moving) / np.sqrt(self.var_moving + self.epsilon)
            self.output = self.inputs_norm * self.weights.reshape(1, c_in, 1, 1)
            self.output += self.biases.reshape(1, c_in, 1, 1)
            return
        
        self.inputs_norm = (inputs - self.mean) / np.sqrt(self.var + self.epsilon)
        self.output = self.inputs_norm * self.weights.reshape(1, c_in, 1, 1)
        self.output += self.biases.reshape(1, c_in, 1, 1)
        
    def backward(self, dvalues):
        '''
        Backward Pass of BatchNorm2D Layer
        
        :param dvalues: (numpy.ndarray) derivative of the loss function with respect 
            to the output; size: (N, C', H', W') where N is the batch size, C' is the 
            output channels, and (H', W') is the output height and width
        
        :returns VOID
        
        '''
        self.dweights = np.sum(dvalues * self.inputs_norm, axis = (0, 2, 3)).reshape([self.n_channels,])
        self.dbiases = np.sum(dvalues, axis = (0, 2, 3)).reshape([self.n_channels,])
        
        N = dvalues.shape[0]
        std = np.sqrt(self.var + self.epsilon)
        dx_norm = dvalues * self.weights.reshape(1, self.n_channels, 1, 1)
        dx_centered = dx_norm / std
        dmean = -(dx_centered.sum(axis=(0, 2, 3)) + 2/N * \
                  self.inputs_norm.sum(axis=(0, 2, 3))).reshape(1, self.n_channels, 1, 1)
        dstd = (dx_norm * self.inputs_norm * -std**(-2)).sum(axis=(0, 2, 3)).reshape(1, self.n_channels, 1, 1)
        dvar = dstd / 2 / std
        
        self.dinputs = dx_centered + (dmean + dvar * 2 * self.inputs_norm) / N        

class Input(Layer):
    ''' 
    Input Layer
    
    Args: None
    
    Notes:
        - Helper layer for Model class
        - Not needed for NPDL users
    
    '''
    def __init__(self):
        super().__init__("input", False)
        
    def forward(self, inputs, training):
        '''
        Forward Pass of Input Layer
        
        :param inputs: (numpy.ndarray) input array of size (N, k) for Linear layers or (N, C, H, W) where N is 
             the batch size, k is the number of neurons in the previous layer, C is the number of channels and 
             (H, W) is the height and width
        
        :param training: (bool) indicates whether the layer is currently training or not
        
        :returns VOID
        
        '''
        self.output = inputs