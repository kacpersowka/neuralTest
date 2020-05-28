from neural import *
import sys,time,pickle

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
w,b=train(x,y,[16,8],n,e,m,functions=function1,lossFunction=[crossEntropy,crossEntropyGradient],functionDerivatives=function1Derivative,functArguments=[[[rectLinear],[softmax]],[[rectLinearDerivative],[softmaxDerivative]]])
for i in range(len(x)):
    print('Input: ',x[i],'\nOutput: ',feedForward(x[i],w,b,y[i],function1,crossEntropy,[[rectLinear],[softmax]])[-2],'\nTarget:',y[i],'\nError:',feedForward(x[i],w,b,y[i],function1,crossEntropy,[[rectLinear],[softmax]])[-1])
    
print("Rounded output:")
for i in range(len(x)):
    print('Input: ',numpy.round(x[i],3),'\nOutput: ',numpy.round(feedForward(x[i],w,b,y[i],function1,crossEntropy,[[rectLinear],[softmax]])[-2],3),'\nTarget:',numpy.round(y[i],3),'\nError:',numpy.round(feedForward(x[i],w,b,y[i],function1,crossEntropy,[[rectLinear],[softmax]])[-1],3))
    
with open("trained.pkl", "bw") as fh:
    data = (w,b)
    pickle.dump(data, fh)