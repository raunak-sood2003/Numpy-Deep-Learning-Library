# NPDL: Numpy Deep Learning Library

![image info](images/npdl_logo.png)

### This repository contains the source code and example notebooks for the NPDL library. 

## About NPDL

NPDL is a deep learning framework written purely in Numpy. I made this
library as an excercise in deep learning theory as well as practice with
Python and Numpy implementation.

### Properties of NPDL

-   *Style*: It has a style that is a mix of
    Keras and PyTorch. While I like the simplicity of Keras, I prefer
    having more control over the training loops as it gives better
    intuition on how neural networks work.
-   *Speed*: The implementations of layers such
    as 2D convolutions and max-pooling are vectorized and perform much
    faster than the traditional nested for-loop implementation.

## Using NPDL

Building the model architecture is very similar to Keras in that we set
up a base model (like Sequential) and add layers/activations to the
model.

For example, the **Sequential** model:

``` python
from npdl.nn.models import Sequential

model = Sequential()
```

Adding `*layers* and *activations* via the **.add()** function.

``` python
from npdl.nn.layers import Linear
from npdl.nn.activations import ReLU

model.add(Linear(64, 10))
model.add(ReLU())
```

Once you are done building the model, you can finish by setting the
*loss* and `*optimizer* functions using the **.finalize()** function.
*Note*: you must create objects for the loss function for the training step later. 
One of the important features of NPDL is that the users must implement the 
training function themselves.

``` python
from npdl.nn.losses import CCELoss
from npdl.nn.optimizers import Adam

loss_fn = CCELoss()
optim_fn = Adam(lr = 1e-3, lr_decay = 0.1)

model.finalize(loss = loss_fn, optimizer = optim_fn)
```

Now we are ready to train the model. You will have to split your data
into batches before this step. An example function for this is given in
the MNIST example notebook. Note that the training functions will vary
based on the task, but the general steps for training are as follows:

``` python
for batchX, batchY in data: # Iterating through training examples and their corresponding targets
    
    y_pred = model.forward(batchX) # Forward pass through the mini-batch
    
    loss, reg_loss = loss_fn.calculate(y_pred, batchY) # Loss calculation
    
    model.backward(y_pred, batchY) # Backwards Pass: computing parameter gradients
    
    model.update_params() # Backwards Pass: updating parameters via gradient descent (or some variation)
```

Finally, to predict on a test set, we can simply run the forward pass of
the model on some examples.

``` python
model.forward(test_example)
```

## Installation

First make sure that you have the latest versions of
`**pip**, `**Python** and **Numpy** installed. Then, in terminal run the
following command:

``` python
pip install -i https://test.pypi.org/simple/ npdl
```

## Next Steps

The library is still in progress as there are many more features that I
plan on adding. Some of which are:

-   *RNN Layers*: Simple RNN cell, LSTM, GRU and
    Embedding layers
-   *Adding loss functions*: i.e KL-divergence
-   `*Adding activations*: i.e PrELU

## Contributions

I hope that the user finds this library easy to use and intutive. I
would love feedback on the libraries functionality, so please feel free
to email me at <rrsood003@gmail.com> for potential improvements. Have
fun builidng your models!
