import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    D = W.shape[0]
    C = W.shape[1]
    N = X.shape[0]

    raw_scores = X.dot(W)
    # doesn't affect softmax calculation, but stops overflow
    raw_scores -= np.max(raw_scores)
    exp_scores = np.exp(raw_scores)
    exp_totals = np.sum(exp_scores, axis=1)
    for i in range(N):
        correct_class = y[i]
        correct_score = exp_scores[i][correct_class]
        loss -= np.log(correct_score / exp_totals[i])

        dW[:, correct_class] -= X[i]
        for i_W in range(D):
            for j_W in range(C):
                dW[i_W][j_W] += X[i][i_W] * exp_scores[i][j_W] / exp_totals[i]

    loss /= N
    dW /= N

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[0]
    C = W.shape[1]
    raw_scores = X.dot(W)
    # doesn't affect softmax calculation, but stops overflow
    raw_scores -= np.max(raw_scores)
    exp_scores = np.exp(raw_scores)
    exp_totals = np.sum(exp_scores, axis=1)

    correct_class_exp_score = exp_scores[np.arange(N), y]
    correct_class_probs = correct_class_exp_score / exp_totals
    loss -= np.sum(np.log(correct_class_probs)) / N
    loss += reg * np.sum(W * W)

    y_onehot = np.zeros((N, C))
    y_onehot[np.arange(N), y] = 1.0
    dW -= X.T.dot(y_onehot)
    probs = exp_scores / exp_totals.reshape((N, 1))
    dW += X.T.dot(probs)
    dW /= N
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
