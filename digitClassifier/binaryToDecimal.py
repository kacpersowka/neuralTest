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
if len(sys.argv)>4:
    l=int(sys.argv[4])
else:
    l=None  
if len(sys.argv)>5:
    t=int(sys.argv[5])
else:
    t=0  
#Seems like neurons further down have a larger chance of being active
random.seed(time.time())
w,b=train(x[t:l],y[t:l],[16,8],n,e,m,functions=function1,lossFunction=[crossEntropy,crossEntropyGradient],functionDerivatives=function1Derivative,functArguments=[[[expRectLinear],[softmax]],[[expRectLinearDerivative],[softmaxDerivative]]])
for i in range(len(x)):
    print('Input: ',x[i],'\nOutput: ',feedForward(x[i],w,b,y[i],function1,crossEntropy,[[expRectLinear],[softmax]])[-2],'\nTarget:',y[i],'\nError:',feedForward(x[i],w,b,y[i],function1,crossEntropy,[[expRectLinear],[softmax]])[-1])
    
print("Rounded output:")
for i in range(len(x)):
    print('Input: ',numpy.round(x[i],3),'\nOutput: ',numpy.round(feedForward(x[i],w,b,y[i],function1,crossEntropy,[[expRectLinear],[softmax]])[-2],3),'\nTarget:',numpy.round(y[i],3),'\nError:',numpy.round(feedForward(x[i],w,b,y[i],function1,crossEntropy,[[expRectLinear],[softmax]])[-1],3))
    
with open("trained.pkl", "bw") as fh:
    data = (w,b)
    pickle.dump(data, fh)