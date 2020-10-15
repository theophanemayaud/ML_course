# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
from th.th_costs import *


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


# ***************************************************
# INSERT YOUR CODE HERE
# TODO: Paste your implementation of grid_search

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = -1*np.ones((len(w0), len(w1))) #set to -1 as losses can only be positive
    # ***************************************************
    # INSERT YOUR CODE HERE
    w0_star, w1_star, loss_star = -1, -1, -1
    for w0_i in range(w0.shape[0]):
        for w1_i in range(w1.shape[0]):
            losses[w0_i][w1_i] = compute_loss(y, tx, np.array([w0[w0_i],w1[w1_i]]).T)
            if loss_star == -1 or loss_star > losses[w0_i][w1_i] :
                loss_star = losses[w0_i][w1_i]
                w0_star = w0[w0_i]
                w1_star = w1[w1_i]
    #print(loss_star, w0_star, w1_star)
    # TODO: compute loss for each combination of w0 and w1.
    # ***************************************************
    #raise NotImplementedError
    return losses

#       here when it is done.
# ***************************************************


