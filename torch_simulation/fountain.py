import torch
import random
import numpy as np
import matplotlib.pyplot as plt


# Custom module to implement Linear transformation scaled by sum of weights
class constrainedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, eps=1e-4):
        """
        Instantiate a basic linear module to hold weight values
        """
        super(constrainedLinear, self).__init__()

        # Instaniate linear module and initialize weights to be all positive
        self.linear = torch.nn.Linear(in_features, out_features)
        self.linear.weight.data = torch.abs(self.linear.weight.data)

        # Term to prevent divide by 0 errors during scaling
        self.eps = eps


    def forward(self, x):
        """
        Forward function involves performing a basic linear transformation,
        then scaling the result by the sum of the weights for each row of the
        weight matrix.
        """

        # Perform linear transformation of x by weights
        linear = self.linear(x)

        # Sum the outgoing weights of each node and divide to get scaled result
        sum = torch.sum(self.linear.weight.data,1) + self.eps
        y = torch.div(linear,sum)
        return y


# Custom function to scale the weights in the range [0,1]
class WeightClipper(object):

    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        # Apply to weights only
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0,1)
            module.weight.data = w

"""
" Test the fountain model with given hyper parameters. This trains
" for 500 epochs.
"
" @param N
" @param depth
" @param D_in
" @param H_1
" @param H_2
" @param D_out
" @param learning_rate
" 
" @returns test_accuracy
"
"""
def TestModel(N, depth, D_in, H_1, H_2, D_out, learning_rate):

    # Construct our model by instantiating the class defined above
    if(depth == 1):
        model = torch.nn.Sequential(
            constrainedLinear(D_in, H_1),
            torch.nn.ReLU(),
            constrainedLinear(H_1, D_out)
        )
    elif(depth == 2):
        model = torch.nn.Sequential(
            constrainedLinear(D_in, H_1),
            torch.nn.ReLU(),
            constrainedLinear(H_1, H_2),
            torch.nn.ReLU(),
            constrainedLinear(H_2, D_out),
        )

    # Apply weight constraint
    clipper = WeightClipper()
    model.apply(clipper)

    # Generate 1000 random datapoints from 2 different 2D Gaussian distributions
    n1 = torch.distributions.Normal(torch.tensor([1.0,2.0]), torch.tensor([0.6,0.2]))
    n2 = torch.distributions.Normal(torch.tensor([1.0,4.0]), torch.tensor([0.9,1.3]))

    x1 = n1.sample((500,))
    x2 = n2.sample((500,))

    X = torch.cat((x1,x2),0)
    Y = torch.cat((torch.zeros((500,1)),torch.ones((500,1))),0)

    #Shuffle dataset and split into train and test sets with 80/20 split
    r = torch.randperm(1000)
    X = X[r]
    Y = Y[r]

    X_train,Y_train = X[:800],Y[:800]
    X_test,Y_test = X[800:],Y[800:]

    # Initialize loss function (Mean squared error) and optimizer (vanilla
    # stochastic gradient descent)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    # Run 500 epochs (probably way too many)
    for i in range(500):

        # Run each batch
        for j in range(80):

            # Compute forward pass
            y_pred = model(X_train[j*N:(j+1)*N])

            # Compute and print loss
            loss = criterion(y_pred, Y_train[j*N:(j+1)*N])

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save and print train and test losses for each epoch
        y_predicted = model(X_test)
        test_losses.append(criterion(y_predicted, Y_test))
        y_predicted = model(X_train)
        train_losses.append(criterion(y_predicted, Y_train))

        #print("Epoch: %d\tTrain Loss = %f\tTest Loss = %f"%(i, train_losses[i].item(), test_losses[i].item()))

    # Evaluate test set and print accuracy
    y_predicted = model(X_test)
    test_accuracy = (torch.sum(torch.eq((y_predicted > 0.5).double(),Y_test.double())).data.numpy())/200.
    print "Test accuracy:   ", test_accuracy, "\n"
    #Return the test accuracy to the parent call
    return test_accuracy