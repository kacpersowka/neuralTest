from neural import *
import sys,time
import pickle
import matplotlib.pyplot as plt

with open("trained.pkl", "br") as fh:
    w,b = pickle.load(fh)

"""    
x=[[0.0,0.0,0.0],
   [0.0,0.0,1.0],
   [0.0,1.0,0.0],
   [0.0,1.0,1.0],
   [1.0,0.0,0.0],
   [1.0,0.0,1.0],
   [1.0,1.0,0.0],
   [1.0,1.0,1.0]]
y=[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
   [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
   [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
   [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],
   [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],
   [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
   [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]]
"""
x=[[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
y=[[0.0,1.0],[1.0,0.0],[1.0,0.0],[0.0,1.0]]
w,b=generateRandomWeightsAndBiases([2,2,2])
a=feedForward(x[0],w,b,y[0],function1,crossEntropy,[[expRectLinear],[softmax]])
g=backProp(w,b,a,y[0],function1Derivative,crossEntropyGradient,[[expRectLinearDerivative],[softmaxDerivative]])
