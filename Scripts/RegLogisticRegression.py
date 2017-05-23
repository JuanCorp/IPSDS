# -*- coding: utf-8 -*-
"""
Created on Mon May 22 11:47:06 2017

@author: jununez
"""

import numpy as np
import time
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

def Polynomial_Features(X):
    print("Fuck")

def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # Get indexes for class 0 and class 1
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    
    # If no specific axes object has been passed, get the current axes.
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True);

class RegLogisticRegression():
    
    def sigmoid_function(self,z):
        return (1/(1 + np.exp(-z) + 1e-8))
    
    def Cost_Function(self,X,y,reg,theta):
        m = y.size
        hyp = self.sigmoid_function(X.dot(theta))
        
        
        p = np.log(hyp).T.dot(y)
        invp = np.log(1-hyp).T.dot(1-y)
        regularizer = (reg/(2*m)) * np.sum(np.square(theta[1:]))
        
        J = -1*(1/m) * (p + invp) + regularizer
        if np.isnan(J[0]):
            return np.inf
        return J[0]
    
    
    def Gradient(self,X,y,reg,theta):
        m = y.size
        hyp = self.sigmoid_function(X.dot(theta.reshape(-1,1)))
        total = 1/m
        regularizer = (reg/m) * np.r_[[[0]],theta[1:].reshape(-1,1)]
        
        G = total * X.T.dot(hyp - y) + regularizer
        
        return G.flatten()
    
    def predict(self,theta, X, threshold=0.5):
        p = self.sigmoid_function(X.dot(theta.T)) >= threshold
        return(p.astype('int'))

    def OptimizeTheta(self,X,y,reg,initial_theta,lrate = 0.001):
        min_cost = np.inf
        theta = initial_theta
        best_theta = initial_theta
        cost_history = []
        cost_history.append(np.inf)
        cost_counter = 0
        for i in range(0,200000):
            cost = self.Cost_Function(X,y,reg,theta)
            g = self.Gradient(X,y,reg,theta)
            if cost < min_cost:
                min_cost = cost
                best_theta = theta
            theta = theta - (lrate * g)
            if(i%10000 == 0):
                print("Cost is now :" + str(cost))
                cost_history.append(cost)
                cost_counter+=1
                if cost_history[cost_counter] > cost_history[cost_counter-1]:
                    lrate/=2
        return best_theta


    
        
data = np.loadtxt("LogR2.csv",delimiter = ",")
y = np.c_[data[:,2]]
X = data[:,0:2]
poly = PolynomialFeatures(7)
XX = poly.fit_transform(X)

clf = RegLogisticRegression()
initial_theta = np.zeros(XX.shape[1])
losses = []
lambdas = [0.1]
lr = [0.1,1,10]
min_loss = np.inf
best_lr = None
best_lam = None
prev_best_loss = np.inf
lr_epochs = 10
lr_range = 50
#finding best lr,lam iteratively.
for lam in lambdas:
  print("training for lambda: " + str(lam) )
  for j in range(lr_epochs):
      for i in range(0,3):
          print("Training for learning rate: " + str(lr[i]))
          best = clf.OptimizeTheta(XX,y,lam,initial_theta,lr[i])
          cur_loss = clf.Cost_Function(XX,y,lam,best)
          losses.append(cur_loss)
          if cur_loss < min_loss:
              min_loss = cur_loss
              best_lam = lam
              best_lr = lr[i]

      print("Best learning rate was: " + str(best_lr))
      if min_loss <= prev_best_loss:
        lr[0] = best_lr - lr_range
        lr[1] = best_lr
        lr[2] = best_lr + lr_range
        lr_range/=2
        prev_best_loss = min_loss
      else:
        break


print(losses)
best_theta = clf.OptimizeTheta(XX,y,best_lam,initial_theta,best_lr)
p = clf.predict(best_theta, XX) 
print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))
print(best_lam)
print(best_lr)
#Plot
# Scatter plot of X,y
plotData(data, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')
    
    # Plot decisionboundary
x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = clf.sigmoid_function(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(best_theta))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');       