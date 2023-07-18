import numpy as np
from npdl.nn.module import Loss

'''
***************************************************** Loss Functions *****************************************************

'''

class CCELoss(Loss):
    '''
    Categorical Cross-Entropy Loss
    
    Equation:
        L(y_pred, y) = -sum(y_i * log(y_pred_i)) | i = 1 to n
    
    Notes:
        - n = number of classes in classification
        - Used in multi-class classification problems
          
    '''
    def forward(self, y_pred, y_true):
        '''
        CCE Loss Forward Pass
        
        :param y_pred: (numpy.ndarray) model's predicted output vector
        :param y_true: (numpy.ndarray) ground truth vector
        
        :returns CCE Loss value
        
        '''
        samples = len(y_pred)
        epsilon = 1e-7
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if y_true.shape[-1] == 1: # Sparse true vector
            preds = y_pred_clipped[range(samples), y_true]
        else: # One hot encoded true vector
            preds = np.sum(y_pred_clipped * y_true, axis = 1)
        
        loss = -np.log(preds)
        return loss
    
    def backward(self, dvalues, y_true):
        '''
        CCE Loss Backwards Pass
        
        :param dvalues: (numpy.ndarray) derivative of loss function 
            with respect to predicted output
        
        :param y_true: (numpy.ndarray) ground truth vector
        
        :returns VOID
        
        '''
        if len(y_true.shape) == 1:
            y_true = np.eye(len(dvalues[0]))[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs /= len(dvalues)

class BCELoss(Loss):
    '''
    Binary Cross-Entropy Loss
    
    Equation:
        L(y_pred, y) = - (1 / N) * sum(y_i * log(y_pred_i) + (1 - y_i) * log(1 - y_pred_i)) | i = 1 to N
    
    Notes:
        - N = number of examples in the batch
        - Used in binary classification problems
          
    '''
    def forward(self, y_pred, y_true):
        '''
        BCE Loss Forward Pass
        
        :param y_pred: (numpy.ndarray) model's predicted output vector
        :param y_true: (numpy.ndarray) ground truth vector
        
        :returns BCE Loss value
        
        '''
        epsilon = 1e-7
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_loss = np.mean(loss, axis = -1)
        return sample_loss
    
    def backward(self, dvalues, y_true):
        '''
        BCE Loss Backwards Pass
        
        :param dvalues: (numpy.ndarray) derivative of loss function 
            with respect to predicted output
        
        :param y_true: (numpy.ndarray) ground truth vector
        
        :returns VOID
        
        '''
        epsilon = 1e-7
        dvalues_clipped = np.clip(dvalues, epsilon, 1 - epsilon)
        self.dinputs = -(y_true / dvalues_clipped - (1 - y_true) / (1 - dvalues_clipped))
        self.dinputs /= len(dvalues[0])
        self.dinputs /= len(dvalues)

class MSELoss(Loss):
    '''
    Mean Squared Error Loss
    
    Equation:
        L(y_pred, y) = (1 / N) * sum((y_i - y_pred_i) ^ 2) | i = 1 to N
    
    Notes:
        - N = number of examples in the batch
        - Used in linear regression problems
          
    '''
    def forward(self, y_pred, y_true):
        '''
        MSE Loss Forward Pass
        
        :param y_pred: (numpy.ndarray) model's predicted output vector
        :param y_true: (numpy.ndarray) ground truth vector
        
        :returns MSE Loss value
        
        '''
        loss = np.sum((y_pred - y_true) ** 2, axis = -1)
        return loss
    
    def backward(self, dvalues, y_true):
        '''
        MSE Loss Backwards Pass
        
        :param dvalues: (numpy.ndarray) derivative of loss function 
            with respect to predicted output
        
        :param y_true: (numpy.ndarray) ground truth vector
        
        :returns VOID
        
        '''
        N = dvalues.shape[0]
        self.dinputs = 2 * (dvalues - y_true) / N
        self.dinputs /= len(dvalues)

class MAELoss(Loss):
    '''
    Mean Absolute Error Loss
    
    Equation:
        L(y_pred, y) = (1 / N) * sum(|y_i - y_pred_i|) | i = 1 to N
    
    Notes:
        - N = number of examples in the batch
        - Used in linear regression problems
          
    '''
    def forward(self, y_pred, y_true):
        '''
        MAE Loss Forward Pass
        
        :param y_pred: (numpy.ndarray) model's predicted output vector
        :param y_true: (numpy.ndarray) ground truth vector
        
        :returns MAE Loss value
        
        '''
        loss = np.mean(np.abs(y_pred - y_true), axis = -1)
        return loss
    
    def backward(self, dvalues, y_true):
        '''
        MAE Loss Backwards Pass
        
        :param dvalues: (numpy.ndarray) derivative of loss function 
            with respect to predicted output
        
        :param y_true: (numpy.ndarray) ground truth vector
        
        :returns VOID
        
        '''
        N = dvalues.shape[0]
        self.dinputs = np.sign(dvalues - y_true) / N
        self.dinputs /= len(dvalues)

        
'''
Notes:
    - The implementation of CCE and BCE loss clips values of 
      y_pred between epsilon and 1 - epsilon so that we don't 
      run into issues for logs of non-positive values
    - All loss functions inherit regularization features from 
      global Loss class in npdl.nn.modules
'''
