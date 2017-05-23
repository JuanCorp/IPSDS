# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:05:50 2017

@author: jununez
"""

import numpy as np
import time
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize




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

class LogisticRegression():
    
    def sigmoid_function(self,z):
        return (1/(1 + np.exp(-z) + 1e-8))
    
    def Cost_Function(self,X,y,theta):
        m = y.size
        hyp = self.sigmoid_function(X.dot(theta))
        
        p = np.log(hyp).T.dot(y)
        invp = np.log(1-hyp).T.dot(1-y)
        
        J = -1*(1/m) * (p + invp)
        if np.isnan(J[0]):
            return np.inf
        return J[0]
    
    
    def Gradient(self,X,y,theta):
        m = y.size
        hyp = self.sigmoid_function(X.dot(theta.reshape(-1,1)))
        total = 1/m
        
        
        G = total * X.T.dot(hyp - y)
        
        return G.flatten()
    
    def predict(self,theta, X, threshold=0.5):
        p = self.sigmoid_function(X.dot(theta.T)) >= threshold
        return(p.astype('int'))

    def OptimizeTheta(self,X,y,initial_theta,lrate = 0.001):
        min_cost = np.inf
        theta = initial_theta
        best_theta = initial_theta
        cost_history = []
        cost_history.append(np.inf)
        cost_counter = 0
        for i in range(0,200000):
            cost = self.Cost_Function(X,y,theta)
            g = self.Gradient(X,y,theta)
            if cost < min_cost:
                min_cost = cost
                best_theta = theta
            theta = theta - (lrate * g)
            if(i%10000 == 0):
                print("Cost is now :" + str(cost))
                print("Best theta is now:" + str(theta))
                time.sleep(1)
                cost_history.append(cost)
                cost_counter+=1
                if cost_history[cost_counter] > cost_history[cost_counter-1]:
                    lrate/=2
        return best_theta


    
        
data = np.loadtxt("LogR.csv",delimiter = ",")
X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = np.c_[data[:,2]]

print(X)

clf = LogisticRegression()
initial_theta = np.zeros(X.shape[1])
cost = clf.Cost_Function(X,y,initial_theta)
gradient = clf.Gradient(X,y,initial_theta)

print(cost)
print(gradient)
losses = []
lr = [0.0163,0.0166,0.0169,0.0172,0.0175]
min_loss = np.inf
best_lr = None
#finding best lr, iteratively.

for l in lr:
    print("Training for learning rate: " + str(l))
    best = clf.OptimizeTheta(X,y,initial_theta,l)
    cur_loss = clf.Cost_Function(X,y,best)
    losses.append(cur_loss)
    if cur_loss < min_loss:
        best_lr = l

best_theta = clf.OptimizeTheta(X,y,initial_theta,best_lr)
p = clf.predict(best_theta, X) 
print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))

#Plot
plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = clf.sigmoid_function(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(best_theta))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');


    