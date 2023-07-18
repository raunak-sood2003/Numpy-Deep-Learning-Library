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
-   `<u>`{=html}Speed`</u>`{=html}: The implementations of layers such
    as 2D convolutions and max-pooling are vectorized and perform much
    faster than the traditional nested for-loop implementation.
:::

::: {#c85ceb9d .cell .markdown}
## Using NPDL

Building the model architecture is very similar to Keras in that we set
up a base model (like Sequential) and add layers/activations to the
model.
:::

::: {#2f3f9943 .cell .markdown}
For example, the **Sequential** model:
:::

::: {#5134993c .cell .code execution_count="1"}
``` python
from npdl.nn.models import Sequential

model = Sequential()
```
:::

::: {#3d8089db .cell .markdown}
Adding `<u>`{=html}layers`</u>`{=html} and
`<u>`{=html}activations`</u>`{=html} via the **.add()** function.
:::

::: {#a3a2b407 .cell .code execution_count="3"}
``` python
from npdl.nn.layers import Linear
from npdl.nn.activations import ReLU

model.add(Linear(64, 10))
model.add(ReLU())
```
:::

::: {#58d70fde .cell .markdown}
Once you are done building the model, you can finish by setting the
`<u>`{=html}loss`</u>`{=html} and `<u>`{=html}optimizer`</u>`{=html}
functions using the **.finalize()** function.
`<u>`{=html}Note`</u>`{=html}: you must create objects for the loss
function for the training step later. One of the important features of
NPDL is that the users must implement the training function themselves.
:::

::: {#4e41a09d .cell .code execution_count="4"}
``` python
from npdl.nn.losses import CCELoss
from npdl.nn.optimizers import Adam

loss_fn = CCELoss()
optim_fn = Adam(lr = 1e-3, lr_decay = 0.1)

model.finalize(loss = loss_fn, optimizer = optim_fn)
```
:::

::: {#681791c1 .cell .markdown}
Now we are ready to train the model. You will have to split your data
into batches before this step. An example function for this is given in
the MNIST example notebook. Note that the training functions will vary
based on the task, but the general steps for training are as follows:
:::

::: {#230fb80a .cell .code}
``` python
for batchX, batchY in data: # Iterating through training examples and their corresponding targets
    
    y_pred = model.forward(batchX) # Forward pass through the mini-batch
    
    loss, reg_loss = loss_fn.calculate(y_pred, batchY) # Loss calculation
    
    model.backward(y_pred, batchY) # Backwards Pass: computing parameter gradients
    
    model.update_params() # Backwards Pass: updating parameters via gradient descent (or some variation)
```
:::

::: {#b59af4aa .cell .markdown}
Finally, to predict on a test set, we can simply run the forward pass of
the model on some examples.
:::

::: {#4caf6475 .cell .code}
``` python
model.forward(test_example)
```
:::

::: {#43cc2dfa .cell .markdown}
## Installation
:::

::: {#e2b9b5ce .cell .markdown}
First make sure that you have the latest versions of
`<u>`{=html}pip`</u>`{=html}, `<u>`{=html}Python`</u>`{=html} and
`<u>`{=html}Numpy`</u>`{=html} installed. Then, in terminal run the
following command:
:::

::: {#53068c23 .cell .code}
``` python
pip install -i https://test.pypi.org/simple/ npdl
```
:::

::: {#e865daef .cell .markdown}
## Next Steps

The library is still in progress as there are many more features that I
plan on adding. Some of which are:

-   `<u>`{=html}RNN Layers`</u>`{=html}: Simple RNN cell, LSTM, GRU and
    Embedding layers
-   `<u>`{=html}Adding loss functions`</u>`{=html}: i.e KL-divergence
-   `<u>`{=html}Adding activations`</u>`{=html}: i.e PrELU
:::

::: {#d3e12bfe .cell .markdown}
## Contributions

I hope that the user finds this library easy to use and intutive. I
would love feedback on the libraries functionality, so please feel free
to email me at <rrsood003@gmail.com> for potential improvements. Have
fun builidng your models!
:::
