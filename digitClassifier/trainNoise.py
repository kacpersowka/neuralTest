from neural import *
import sys,time
import pickle,random

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
y=numpy.append(y,[[0.01] for k in range(len(y))],axis=1)
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

with open("noise.pkl", "br") as fh:
    data = pickle.load(fh)
x=numpy.append(x,data[0])
y=numpy.append(y,data[1])
    
#train n e m bs lf l
#train e bs lf l n m
if len(sys.argv)>1:
    e=float(sys.argv[1])
else:
    e=0.1
if len(sys.argv)>2:
    bs=int(sys.argv[2])
else:
    bs=5
if len(sys.argv)>3:
    lf=float(sys.argv[3])
else:
    lf=1.001
if len(sys.argv)>4:
    l=int(sys.argv[4])
else:
    l=None 
if len(sys.argv)>5:
    n=int(sys.argv[5])
else:
    n=1
if len(sys.argv)>6:
    m=float(sys.argv[6])
else:
    m=0.8
if len(sys.argv)>7:
    wf=int(sys.argv[7])
else:
    wf=1
if len(sys.argv)>8:
    bf=int(sys.argv[8])
else:
    bf=1

random.seed(time.time())
w,b=sgd(x[:l],y[:l],[8,8,11],n,e,m,wf,bf,bs,lf,functions=function1,lossFunction=[crossEntropy,crossEntropyGradient],functionDerivatives=function1Derivative,functArguments=[[[rectLinear],[softmax]],[[rectLinearDerivative],[softmaxDerivative]]])
with open("trained.pkl", "bw") as fh:
    data = (w,b)
    pickle.dump(data, fh)
