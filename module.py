import numpy as np

class Layer:
    '''
    Global Layer Class
    
    Args:
        layer_name - name of the layer/activation
        trainable - boolean indicating whether 
        the layer is trainable or not
    
    Notes:
        - Gives each layer/activation a name 
          and boolean trainable param
    '''
    def __init__(self, layer_name, trainable):
        self.layer_name = layer_name
        self.trainable = trainable

class Loss:
    '''
    Global Loss Class
    
    Args: None
    
    Notes:
        - Implements regularization loss for all 
          loss functions
    '''
    def remember_reg_layers(self, reg_layers):
        '''
        Helper Function for Model Class (npdl.nn.models)
        
        :param reg_layers: (object list) layers that have regularization parameters
        
        :returns VOID
        '''
        self.reg_layers = reg_layers
    
    def calculate(self, y_pred, y):
        '''
        Loss Function Calculation
        
        :param y_pred: (numpy.ndarray) model's predicted output vector
        :param y_true: (numpy.ndarray) ground truth vector
        
        :returns (float) loss from data and (float) regularization separately 
        '''
        sample_losses = self.forward(y_pred, y)
        data_loss = np.mean(sample_losses)
        return data_loss, self.regularization_loss()
    
    def regularization_loss(self):
        '''
        Computes Regularization Loss
        
        :returns (float) regularization loss
        '''
        reg_loss = 0
        for layer in self.reg_layers:
            reg_loss += layer.weight_reg_l1 * np.sum(np.abs(layer.weights))
            reg_loss += layer.bias_reg_l1 * np.sum(np.abs(layer.biases))
            reg_loss += layer.weight_reg_l2 * np.sum(layer.weights ** 2)
            reg_loss += layer.bias_reg_l2 * np.sum(layer.biases ** 2)
        return reg_loss