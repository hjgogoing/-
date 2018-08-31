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
  num_batch = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)
  scores_exp = np.exp(scores)
  sum_exp1 = np.sum(scores_exp,axis=1)
  sum_exp2 = np.reshape(sum_exp1,[sum_exp1.shape[0],1])
  soft_exp = scores_exp/sum_exp2
  for i in range(num_batch):
    loss += -np.log(soft_exp[i,y[i]])
    for j in range(num_class):
      if j != y[i]:
        dW[:,j] += (scores_exp[i,j]/sum_exp1[i])*X[i]
      if j == y[i]:
        dW[:,j] += ((scores_exp[i,y[i]]/sum_exp1[i]) - 1)*X[i]

  loss /= num_batch
  loss += 0.5*reg*np.sum(W*W)
  dW /= num_batch
  dW += reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #pass
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
  num_batch = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)
  scores_exp = np.exp(scores)
  sum_sco_exp1 = np.sum(scores_exp,axis=1)
  sum_sco_exp2 = np.reshape(sum_sco_exp1,[num_batch,1])
  soft_exp = scores_exp/sum_sco_exp2   # mx10
  a = soft_exp[range(num_batch),y]
  b = -np.log(a)
  loss = np.sum(b)
  #loss = np.sum(-np.log(soft_exp[range[num_batch],y]))
  loss /= num_batch
  loss += 0.5*reg*np.sum(W*W)
  soft_exp[range(num_batch),y] -= 1
  dW = np.dot(X.T,soft_exp)
  dW /= num_batch
  dW += reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def train(W,X,y,num_iter,reg,lr):
  
  loss_history=[]
  for i in range(num_iter):
    loss,dW = softmax_loss_vectorized(W, X, y, reg)
    loss_history.append(loss)
    W -= dW*lr
    
  return W,loss_history

def pred(W,X,y):
  
  y_p = np.zeros([X.shape[0]])
  y_m = np.dot(X,W)
  y_p = np.argmax(y_m,axis=1)
  return y_p
    
    
    