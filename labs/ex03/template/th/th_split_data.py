# -*- coding: utf-8 -*-
""" Splitting data """

import numpy as np

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """    
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    
    if ratio<0 or ratio>1:
        raise NameError("Ratio is out of [0 1] range")

    nb_data_pts = len(y)
    nb_train_pts = int(np.rint(nb_data_pts*ratio))
    
    train_true_false = np.full(nb_data_pts, False)

    train_pts_indexes = np.random.choice(np.arange(start=0, stop=nb_data_pts), size=nb_train_pts, replace=False) # high is actually one above the max possible integer the function might return
    train_true_false[train_pts_indexes] = True
    
    train_x = x[train_true_false]
    train_y = y[train_true_false]
    
    test_x = x[~train_true_false]
    test_y = y[~train_true_false]
    
    return train_x, train_y, test_x, test_y
    
    # ***************************************************
    #raise NotImplementedError
    
# THEO test my code : 
#split_data(x=np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]]).T, y=np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).T, ratio=0.7)