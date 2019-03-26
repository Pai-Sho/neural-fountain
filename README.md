# neural-fountain
An analog (electricity free) physical neural network implemented using water flow and buckets instead of weights and nodes.

## TODO

* What task will the neural network perform.

* How does one represent the training data. Ideally we want our training data to be high-dimension. If we use water to measure an object as input, we can only have data with dimension 2: volume and mass.

* How does the forward pass work?

* How does the backward pass work?

* Once the above things have been determined: Implement a simple neural net in tensorflow, use tensorboard to collect data on its training characteristics. This could help because we want a network that trains fast, and has low complexity. Before building a physical neural network we will want to optimize hyperparameters such as batch size, learning rate, and neural network topology. 


## Ideas so far

* 3D printed "buckets" - shape determines activation function
* Valve adjustments to implement weights/training
* implement in tensorflow
* Simulate in blender
* Network is tilted so that the output is lower than the input. When ALL the water from a training example reaches the output nodes it will trigger a balance that tilts the neural network so that the input is lower than the output, thus implementing a feed forward. If we trigger this balance by another weight it would represent batch size. (If a single training example uses 100ml of water a batch size of 100 would have 100*100ml of water).