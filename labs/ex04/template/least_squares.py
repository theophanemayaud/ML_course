# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """ calculate the least squares solution.
    
    Parameters
    ----------
    y : the vector of answer values 
    tx : the matrix of input values with augmented 
        first column of 1 to account for constant w0 parameter. 
        Rows are datapoints of D dimensions, 
        and columns are features of N dimensions
    out : mse, w
    """
                    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    
    # There are two methods (both tested and working)
    
    # 1. Manual method with inverse
    # w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx)), tx.T), y)

    # 2. With np solving method, solves x in Ax=b, here we have XˆT*X*w_star = XˆT*y => More robust method because works even if not inversible.
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    
    # returns mse, and optimal weights
    mse = compute_loss(y, tx, w)
    return mse, w
    
