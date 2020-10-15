# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    loss = compute_loss(y, tx, w)
    #print(loss)
    gradient = compute_gradient(y, tx, w)    
    w = w - gamma*gradient
    return w
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    #raise NotImplementedError


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    ws = [initial_w]
    losses = []
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=max_iters):

        loss = compute_loss(minibatch_y, minibatch_tx, w)

        gradient = compute_gradient(minibatch_y, minibatch_tx, w)

        w = w - gamma*gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent: loss={l}, w0={w0}, w1={w1}".format(l=loss, w0=w[0], w1=w[1]))
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    #raise NotImplementedError
    return losses, ws