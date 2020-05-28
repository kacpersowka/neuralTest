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
x=train_imgs
y=train_labels_one_hot
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
if len(sys.argv)>3:
    m=float(sys.argv[3])
else:
    m=0.8   
random.seed(time.time())
w,b=train(x,y,[8,8,10],n,e,m,functions=function1,lossFunction=[crossEntropy,crossEntropyGradient],functionDerivatives=function1Derivative,functArguments=[[[rectLinear],[softmax]],[[rectLinearDerivative],[softmaxDerivative]]])
with open("trained.pkl", "bw") as fh:
    data = (w,b)
    pickle.dump(data, fh)
