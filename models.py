import numpy as np
import copy
import pickle
from npdl.nn.layers import Input

'''
************************************************* Sequential Model API *************************************************

'''

class Sequential:
    '''     
    Args: None
    
    Methods:
        add - add layers to the neural network model
        finalize - compiles model and recognizes loss and optimizer functions given by the user
        forward - computes the forward pass of the data
        backward - computes the backward pass (gradient calculation)
        update_params - uses the optimizer to update the model params
        save - saves model to a path as a .model file
        load - loads model given the path name
    
    Notes:
        - For loss calculation, user must use a loss function (from losses) to compare
          y_pred and y_true
        - Users must write their own training function for training over several epochs
    
    '''
    def __init__(self):
        self.input_layer = Input()
        self.layers = [self.input_layer]
    
    def add(self, layer):
        '''
        Model Add Function for Layers
        
        :param layer: layer object from npdl.nn.layers
        
        :returns VOID
        '''
        self.layers.append(layer)
    
    def finalize(self, *, loss, optimizer):
        '''
        Model Compilation Function
        
        :param loss: loss function object from npdl.nn.losses
        :param optimizer: optimizer function object from npdl.nn.optimizers
        
        :returns VOID
        
        '''
        self.loss = loss
        self.optimizer = optimizer
        self.n_layers = len(self.layers)
        self.reg_layers = []
        
        for i in range(self.n_layers - 1):
            self.layers[i].next = self.layers[i+1]
            if self.layers[i].layer_name == 'linear' or self.layers[i].layer_name == 'Conv2D':
                self.reg_layers.append(self.layers[i])
        
        self.layers[-1].next = self.loss
        if hasattr(self.layers[-1], 'weights'):
            self.reg_layers.append(self.layers[-1])
        
        for i in range(1, self.n_layers):
            self.layers[i].prev = self.layers[i-1]
        
        self.loss.remember_reg_layers(self.reg_layers)
        
    def forward(self, inputs, training):
        '''
        Forward Method for Entire Model
        
        :param inputs: (numpy.ndarray) input vector to model
        :param training: (bool) indicates whether model is in training phase or not
        
        :returns: (numpy.ndarray) output vector of model
        '''
        for i in range(self.n_layers):
            if i == 0:
                self.layers[i].forward(inputs, training)
            else:
                self.layers[i].forward(self.layers[i].prev.output, training)
        
        return self.layers[-1].output
    
    def backward(self, y_pred, y_true):
        '''
        Backward Method (computes gradients) For Entire Model
        
        :param y_pred: (numpy.ndarray) model's predicted output vector
        :param y_true: (numpy.ndarray) ground truth vector
        
        :returns VOID
        
        '''
        self.loss.backward(y_pred, y_true)
        for i in range(self.n_layers - 1, 0, -1):
            self.layers[i].backward(self.layers[i].next.dinputs)
    
    def update_params(self):
        '''
        Update Model Params
        
        :returns VOID
        
        '''
        self.optimizer.pre_update_params()
        for layer in self.layers:
            if layer.trainable:
                self.optimizer.update_params(layer)
        self.optimizer.post_update_params()
        
    def save(self, path):
        '''
        Saves Model Weights to Directory
        
        :param path: (str) path to desired file location
        
        :returns VOID
        
        '''
        model = copy.deepcopy(self)
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        
        layer_props = ['inputs', 'output', 'dinputs', 'dweights', 'dbiases', 'dgamma', 'dbeta']
        
        for layer in model.layers:
            for prop in layer_props:
                if hasattr(layer, prop):
                    layer.__dict__.pop(prop, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load(path):
        '''
        Loads Model Weights
        
        :param path: (str) path to model file location
        
        :returns model object
        
        '''
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model