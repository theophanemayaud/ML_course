# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np

def compute_loss(y, tx, w):
    """Calculate the mean squared error loss.
    
    Return
    ------
    mse : mean squared error, noramlize to RMSE with np.sqrt(2*mse) 
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    mse = np.dot( (y-np.dot(tx, w)).T, y-np.dot(tx, w))/ (2*y.shape[0])

    return mse
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    #raise NotImplementedError
    

def compute_rmse(y, tx, w):
    """Calculate the root mean squared error (standarized loss).
    
    Return
    ------
    rmse : root mean squared error.
    """
    mse = compute_loss(y, tx, w)
    rmse = np.sqrt(2*mse)

    return rmse