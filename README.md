# mlp_mnist:

### Purpose:
The purpose of this repository is to perform a parametric study and experiment with the modification of gradient decent parameters to better understand the effects on training and testing accuracy. This project uses the MNIST dataset and a `keras` sequential model to train a multi-layered perceptron network for categorical classification.

The work done for this project was a final exam requirement for CSC6240 Mathematics and Theory of Machine Learning, Fall 2020, Tennessee Technological University, Computer Science Department. Here is our [video presentation](https://www.youtube.com/watch?v=6I37zqV2eYg)


### Team:

| Name               |     Email                        | Profile                                                                                       |
|:------------------:|:--------------------------------:|:---------------------------------------------------------------------------------------------:|
| Jonathan Gibson | jagibson44@tntech.edu   | [<img src=".pics/jagibson44.jpeg" width="40px"/>](https://github.com/Lexxeous)                            |
| Ryan Adamson       | radamson42@tntech.edu   | [<img src=".pics/radamson42.png" width="40px"/>](https://github.com/rmadamson)                         |
| John Brackins      | jtbrackins42@tntech.edu | [<img src=".pics/jtbrackins42.png" width="40px"/>](https://github.com/brackins-john)                   |
| Ahsan Ayub         | mayub42@tntech.edu      | [<img src=".pics/mayub42.jpeg" width="40px"/>](https://github.com/AhsanAyub)                           |


### Usage:
Use `results.csv` to log the results of experimental runs.

**Only modify the parameters that are specified:**<br>

> All of the following parameters can be used as command line arguments to the `mlp_mnist.py` script from argument 1 to argument 9.

  * `AUTO_LOG` represents automatically logging data from `mlp_mnist.py` or not. Default value is "1".<br>
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
populate it with required python modules.  To load the `venv` python environment, you should run `source ./venv/bin/activate`.


### References:
MNIST datasets downloaded from:<br>
http://yann.lecun.com/exdb/mnist/

Python code derived from:<br>
https://towardsdatascience.com/introduction-to-multilayer-neural-networks-with-tensorflows-keras-api-abf4f813959
