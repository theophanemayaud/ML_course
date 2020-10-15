# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Output
    ------
    phi_x : matrix formed of augmented features, 
        from x = [x1, x2, x3...].T it returns
        phi_x = [[1, x1, x1ˆ2, x1ˆ3, ..., x1ˆdegree],
                 [1, x2, x2ˆ2, x2ˆ3, ..., x2^degree],
                 ...]
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    
    phi_x = np.empty(shape=(len(x), degree+1))
    for feature_i in range(len(x)): # Using for loop but could probably be done faster with list comprehensions and direct vector power elevation
        for degree_i in range(degree+1):
            phi_x[feature_i][degree_i] = np.power(x[feature_i], degree_i)
            
    return phi_x

    # ***************************************************
    #raise NotImplementedError
