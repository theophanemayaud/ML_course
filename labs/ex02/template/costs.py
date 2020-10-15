# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    mse = np.dot( (y-np.dot(tx, w)).T, y-np.dot(tx, w))/ (2*y.shape[0])

    return mse
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    #raise NotImplementedError