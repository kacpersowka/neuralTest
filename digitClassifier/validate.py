from neural import *
import sys,time
import pickle
import matplotlib.pyplot as plt

with open("pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
x=test_imgs
y=test_labels_one_hot

if len(sys.argv)>1:
    n=int(sys.argv[1])
else:
    n=-1
    
if len(sys.argv)>2:
    offset=int(sys.argv[2])
else:
    offset=0

if len(sys.argv)>3:
    r=int(sys.argv[3])
else:
    r=10    
    
if n==-1:
    n=len(x)
    
with open("trained.pkl", "br") as fh:
    w,b = pickle.load(fh)

for i in range(offset,min(n+offset,len(x))):
    print('Input: ',i,'\nOutput: ',numpy.round(feedForward(x[i],w,b,y[i],function1,mse,[[rectLinear],[softmax]])[-2],r),'\nTarget:',numpy.round(y[i],r),'\nError:',numpy.round(feedForward(x[i],w,b,y[i],function1,mse,[[rectLinear],[softmax]])[-1],r))    
    