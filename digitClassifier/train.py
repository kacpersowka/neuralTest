from neural import *
import sys,time
import pickle

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
    n=int(sys.argv[1])
else:
    n=100
if len(sys.argv)>2:
    e=float(sys.argv[2])
else:
    e=0.1
random.seed(time.time())
w,b=train(train_imgs,train_labels_one_hot,[8,8,10],n,e,functions=function1,lossFunction=[mse,mseGradient],functionDerivatives=function1Derivative,functArguments=[[[rectLinear],[softmax]],[[rectLinearDerivative],[softmaxDerivative]]])
for i in range(len(x)):
    print('Input: ',i,'\nOutput: ',feedForward(x[i],w,b,y[i],function1,mse,[[rectLinear],[softmax]])[-2],'\nTarget:',y[i],'\nError:',feedForward(x[i],w,b,y[i],function1,mse,[[rectLinear],[softmax]])[-1])
    
print("Rounded output:")
for i in range(len(x)):
    print('Input: ',i,'\nOutput: ',numpy.round(feedForward(x[i],w,b,y[i],function1,mse,[[rectLinear],[softmax]])[-2],3),'\nTarget:',numpy.round(y[i],3),'\nError:',numpy.round(feedForward(x[i],w,b,y[i],function1,mse,[[rectLinear],[softmax]])[-1],3))
    
with open("trained.pkl", "bw") as fh:
    data = (w,b)
    pickle.dump(data, fh)
