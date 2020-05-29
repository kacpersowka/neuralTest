from neural import *
import sys,time

x=[[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
y=[[0.0,1.0],[1.0,0.0],[1.0,0.0],[0.0,1.0]]
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
random.seed(time.time())
w,b=train(x[t:l],y[t:l],[6,2],n,e,m,functions=function1,lossFunction=[crossEntropy,crossEntropyGradient],functionDerivatives=function1Derivative,functArguments=[[[expRectLinear],[softmax]],[[expRectLinearDerivative],[softmaxDerivative]]])
for i in range(t,len(x[:l])):
    print(x[i],' : ',feedForward(x[i],w,b,y[i],function1,crossEntropy,[[expRectLinear],[softmax]])[-2],' : ',y[i],' : ',feedForward(x[i],w,b,y[i],function1,crossEntropy,[[expRectLinear],[softmax]])[-1])