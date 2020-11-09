# mlp_mnist:

### Purpose:
The purpose of this repository is to experiment with the modification of stochastic gradient decent parameters to better understand the effects on training and testing accuracy.

### Usage:
Use `results.csv` to log the results of experimental runs.

The `AUTO_RUN` global variable specifies if the `mlp_mnist.py` script automatically logs the relevant data or not.<br>
**Only modify the parameters that are specified:**<br>
  * `TR_ACC` represents the training accuracy metric. This will be calculated automatically. Default value is "None".<br>
  * `TE_ACC` represents the testing accuracy metric. This will be calculated automatically. Default value is "None".<br>
  * `LR` represents represents the learning rate metric. Default value is "0.001".<br>
  * `DECAY` represents the decay metric. Default value is "1e-7".<br>
  * `MOMENTUM` represents the momentum metric. Default value is "0.9".<br>
  * `NESTEROV` represents the usage of Nesterov momentum. Default value is "False".<br>
  * `EPOCHS` represents the number of epochs. Default value is "50".<br>
  * `BATCH_SZ` represents the batch size. Default value is "50".<br>
  * `HIDDEN_DIM` represents the hidden layer dimension size. Default is "50".<br>
  * `NUM_LAYERS` represents the number of dense hidden layers. Manually add another layer if you change this value. Default value is "3".<br>

### Setting up the python environment:
There is a `requirements.txt` file that contains specific versions of libraries used by mlp_mnist.  After checking out this repository, run
`python3 -m venv venv` to create a local python environment directory named `venv`, followed by `venv/bin/pip install -r requirements.txt` to
populate it with required python modules.  To load the `venv` python environment, you should run `./venv/bin/activate`.

### References:

MNIST datasets downloaded from:<br>
http://yann.lecun.com/exdb/mnist/

Python code derived from:<br>
https://towardsdatascience.com/introduction-to-multilayer-neural-networks-with-tensorflows-keras-api-abf4f813959
