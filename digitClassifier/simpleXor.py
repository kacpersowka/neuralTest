from neural import *
import sys,time

x=[[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
y=[[0.0],[1.0],[1.0],[0.0]]
if len(sys.argv)>1:
    n=int(sys.argv[1])
else:
    n=100
if len(sys.argv)>2:
    e=float(sys.argv[2])
else:
    e=0.1
random.seed(time.time())
w,b=train(x,y,[3,1],n,e,0)
for i in range(len(x)):
    print(x[i],' : ',feedForward(x[i],w,b,y[i])[-2],' : ',y[i],' : ',feedForward(x[i],w,b,y[i])[-1])
