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

if len(sys.argv)>1:
    s=int(sys.argv[1])
else:
    s=True
if s:
    x=test_imgs
    y=test_labels_one_hot
    y=numpy.append(y,[[0.01] for k in range(len(y))],axis=1)
    with open("noiseTest.pkl", "br") as fh:
        data = pickle.load(fh)
    x=numpy.append(x,data[0])
    y=numpy.append(y,data[1])
else:
    x=train_imgs
    y=train_labels_one_hot
    y=numpy.append(y,[[0.01] for k in range(len(y))],axis=1)
    with open("noise.pkl", "br") as fh:
        data = pickle.load(fh)
    x=numpy.append(x,data[0])
    y=numpy.append(y,data[1])   
    
if len(sys.argv)>2:
    n=int(sys.argv[2])
else:
    n=-1
    
if len(sys.argv)>3:
    offset=int(sys.argv[3])
else:
    offset=0

if len(sys.argv)>4:
    r=int(sys.argv[4])
else:
    r=3   
    
if n==-1:
    n=len(x)
    
with open("trained.pkl", "br") as fh:
    w,b = pickle.load(fh)

for i in range(offset,min(n+offset,len(x))):
    a=feedForward(x[i],w,b,y[i],function1,crossEntropy,[[expRectLinear],[softmax]])
    print('Input: ',i,'\nOutput: ',numpy.round(a[-2],r),'\nTarget:',numpy.round(y[i],r),'\nError:',numpy.round(a[-1],r),'\nCorrect:',int(a[-2].index(max(a[-2]))==y[i].index(max(y[i]))))    
    