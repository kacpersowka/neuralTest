import numpy
import random

def proper_round(n):
     return int(n)+round(n-int(n)+1)-1

def rectLinear(xx):
    x=numpy.array(xx,dtype=float)
    for i in range(len(x)):
            x[i]=max(0,x[i]) #apply activation function
    return x

def rectLinearDerivative(xx):
    x=numpy.array(xx,dtype=float)
    for i in range(len(x)):
            x[i]=int(x[i]>0) #apply activation function
    return numpy.diag(x)

def expRectLinear(xx,a=0.1):
    x=numpy.array(xx,dtype=float)
    for i in range(len(x)):
        if not (x[i]>0):
            x[i]=a*(numpy.exp(x[i])-1) #apply activation function
    return x

def expRectLinearDerivative(xx,a=0.1):
    x=numpy.array(xx,dtype=float)
    for i in range(len(x)):
        if (x[i]>0):
            x[i]=1
        else:
            x[i]=a*numpy.exp(x[i])
    return numpy.diag(x)

def leakyRectLinear(xx):
    x=numpy.array(xx,dtype=float)
    for i in range(len(x)):
            x[i]=max(0.01,x[i]) #apply activation function
    return x

def leakyRectLinearDerivative(xx):
    x=numpy.array(xx,dtype=float)
    for i in range(len(x)):
            x[i]=max(int(x[i]>0),0.01) #apply activation function
    return numpy.diag(x)

def softmax(x):
    z=numpy.array(x,dtype=float)-max(x)
    return numpy.exp(z) / numpy.sum(numpy.exp(z), axis=0)

def softmaxDerivative(x):
    z=numpy.array(x,dtype=float)-max(x)
    v=numpy.sum(numpy.exp(z), axis=0)
    u=numpy.array([numpy.exp(z)]).transpose()
    du=numpy.diag(u.transpose()[0])
    dv=u.transpose()[0]
    return (numpy.multiply(v,du)-numpy.multiply(u,dv))/v**2

def convolve2d(inp,kernel,stride=1,mode='full'):
    kernel=numpy.array(kernel)
    if mode=='full':
        inp=numpy.pad(inp,len(kernel)-1)
        padOffset=(len(kernel)//2)+1
        startOffset=0
    elif mode=='same':
        inp=numpy.pad(inp,(len(kernel)-1)//2)
        padOffset=((len(kernel)-1)//2)+1
        startOffset=padOffset-2
    elif mode=='valid':
        startOffset=0
        inp=numpy.array(inp)
        padOffset=(len(kernel)//2)+1
    #s=[[0 for a in range(len(inp[0])-padOffset)] for b in range(len(inp)-padOffset)]
    #s=[0 for i in range(((len(inp)-padOffset)*(len(inp[0])-padOffset))//stride)]
    s=[]
    #for k in range(0,((len(inp)-padOffset)//stride[0])*((len(inp[0])-padOffset)//stride[1])):
    #        i=k//(len(inp[0])-padOffset)
    #        j=k%(len(inp[0])-padOffset)
    for i in range(startOffset,(len(inp)-padOffset),stride):
        for j in range(startOffset,(len(inp[0])-padOffset),stride):
            #s[i][j]=sum(numpy.multiply(inp[i:3+i][:,j:3+j],kernel.transpose()).flatten())
            print(numpy.array(inp[i:3+i][:,j:3+j]))
            s.append(sum(numpy.multiply(inp[i:3+i][:,j:3+j],kernel.transpose()).flatten()))
    return numpy.array(s).reshape((proper_round((len(inp)-padOffset)/stride),len(s)//proper_round((len(inp)-padOffset)/stride)))

def crossEntropy(q,p):
    z=numpy.array(q,dtype=float)+1e-7 #For numerical stability
    return -sum(p*numpy.log(z))

def crossEntropyGradient(q,p): #w.r.t q
    z=numpy.array(q,dtype=float)+1e-7
    return [[-p[i]/z[i] for i in range(len(p))]]

def mse(x,y):
    s=0
    for i in range(len(x)):
        s+=(x[i]-y[i])**2
    return s/len(x)

def getPredictions(w,x,act=rectLinear):
    w=numpy.array(w,dtype=float)
    train=numpy.array(x,dtype=float)
    train=train.dot(w)
    for i in range(len(train)):
        train[i]=act(train[i])
    return train

def function1(xx,ww,b,act=rectLinear):
    x=numpy.append(numpy.array(xx,dtype=float),1.0) #add bias column
    w=numpy.array(ww,dtype=float)
    w=numpy.append(w,[[i] for i in b],axis=1) #add biases into weights
    h=w.dot(x) #apply weight and bias to layer
    h=numpy.array(act(h),dtype=float)
    #for i in range(len(h)):
    #        h[i]=act(h[i]) #apply activation function
    return h

def function1Derivative(xx,ww,b,actDer=rectLinearDerivative):
    x=numpy.array(xx,dtype=float)
    w=numpy.array(ww,dtype=float)
    w=numpy.append(w,[[i] for i in b],axis=1) #add biases into weights
    h=w.dot(x) #apply weight and bias to layer
    h=numpy.array(actDer(h))
    #return [numpy.multiply(h.reshape([len(h),-1]),ww),h.reshape([len(h),-1]).dot([x])]
    return [h.dot(ww),numpy.dot(h,[x for i in ww])]

def mseGradient(x,y): #dL/dX (w.r.t predicted)
    return [[2*(x[i]-y[i])/len(x) for i in range(len(x))]]

def backProp(w,b,a,y,functionDer=function1Derivative,lossDerivative=mseGradient,functArguments=[[rectLinearDerivative],[lambda x:[1 for i in x]]]):
    if type(functionDer)!=list:
        functionDer=[functionDer for i in range(len(w))]
    if len(functArguments)==2 and len(w)>2:
        functArguments=functArguments[:-1]+[functArguments[0] for i in range(len(functArguments),len(w))]+functArguments[-1:]
    jacobians=[[] for i in a[:-1]]
    weightGradients=[[] for i in w]
    jacobians.append(numpy.array([1.0])) #manually calculate loss gradient
    jacobians[-2]=numpy.array(lossDerivative(a[-2],y)) #manually calculate prediction gradient
    #grad_table[-2]=numpy.array(grad_table[-2][0])
    for i in range(len(a)-3,-1,-1): #skip loss and prediction
        der=(functionDer[i](a[i],w[i],b[i],*functArguments[i]))
        jacobians[i]=numpy.array(numpy.dot(jacobians[i+1],der[0]))
        weightGradients[i]=numpy.multiply(jacobians[i+1].transpose(),der[1])
    return weightGradients

def generateRandomWeightsAndBiases(l,wf=100,bf=100,weightMap=None,biasMap=None):
    w=[]
    b=[]
    for i in range(1,len(l)):
        if weightMap==None or weightMap[i]==None or weightMap[i]==[]:
            w.append([[random.random()/wf for k in range(l[i-1])] for j in range(l[i])])
        else:
            w.append(weightMap[i])
        if biasMap==None or biasMap[i]==None or biasMap[i]==[]:
            b.append([random.random()/bf for k in range(l[i])])
        else:
            b.append(biasMap[i])

    wb=[]
    for i in range(len(w)): #merge biases into weights
        if (type(b[i])==list or type(b[i])==type(numpy.array(0))):
            wb.append(numpy.append(w[i],[[j] for j in b[i]],axis=1))
        else:
            wb.append(numpy.append(w[i],b[i]))
    return [w,b]

def feedForward(x,w,b,y,functions=function1,lossFunction=mse,functArguments=[[rectLinear],[lambda x:x]]):
    if type(functions)!=list:
        functions=[functions for i in range(len(w))]
    if len(functArguments)==2 and len(w)>2:
        functArguments=functArguments[:-1]+[functArguments[0] for i in range(len(functArguments),len(w))]+functArguments[-1:]
    a=[numpy.array(x)]
    h=x
    for l in range(len(w)):
        h=functions[l](h,w[l],b[l],*functArguments[l])
        if type(h)!=list and type(h)!=type(numpy.array(0)):
            h=[h]
        a.append(numpy.array(h))
    a.append(numpy.array([lossFunction(h,y)]))
    for i in range(len(b)):
        a[i]=numpy.append(a[i],1)
    return a

def updateWeights(w,b,g,e,m=0):
    wb=[]
    for i in range(len(w)): #merge biases into weights
        if (type(b[i])==list or type(b[i])==type(numpy.array(0))):
            wb.append(numpy.append(w[i],[[j] for j in b[i]],axis=1))
        else:
            wb.append(numpy.append(w[i],b[i]))
    #velocity=[[0 for j in range(len(wb[i]))] for i in range(len(wb))]
    velocity=0
    for i in range(len(wb)):
        velocity=m*velocity-e*numpy.array(g)
        wb=wb+velocity
        #for j in range(len(wb[i])):
            #wb[i][j]=wb[i][j]+velocity[i][j]
            #wb[i][j]=wb[i][j]-e*g[i][j]+m*velocity[i][j]
            #velocity[i][j]=e*g[i][j]+m*velocity[i][j]
    #separate weights and biases
    wn=[]
    bn=[]
    for i in range(len(wb)):
        if type(wb[i][0])==type(numpy.array(0)):
            wn.append(numpy.delete(wb[i],-1,axis=1))
            bn.append(wb[i][:,-1])
        else:
            wn.append(numpy.delete(wb[i],-1))
            bn.append(wb[i][-1])
    return [wn,bn]

def train(X,Y,l,n,e,m,init=generateRandomWeightsAndBiases,initArgs=[1,100],functions=function1,lossFunction=[mse,mseGradient],functionDerivatives=function1Derivative,functArguments=[[[rectLinear],[lambda x:x]],[[rectLinearDerivative],[lambda x:numpy.diag([1 for i in x])]]]):
    wn,bn=init([len(X[0])]+l,*initArgs)
    for i in range(n):
        #print('Pass: ',i)
        k=[z for z in range(len(X))]
        random.shuffle(k)
        for j in k:
            #print('Item: ',j)
            x=X[j]
            y=Y[j]
            a=feedForward(x,wn,bn,y,functions,lossFunction[0],functArguments[0])
            g=backProp(wn,bn,a,y,functionDerivatives,lossFunction[1],functArguments[1])
            wn,bn=updateWeights(wn,bn,g,e,m)
    return [wn,bn]

def sgd(X,Y,l,n,e,m,init=generateRandomWeightsAndBiases,initArgs=[1,100],batchSize=5,learningFactor=1.001,functions=function1,lossFunction=[mse,mseGradient],functionDerivatives=function1Derivative,functArguments=[[[rectLinear],[lambda x:x]],[[rectLinearDerivative],[lambda x:numpy.diag([1 for i in x])]]]):
    wn,bn=init([len(X[0])]+l,*initArgs)
    for i in range(n):
        #print('Pass: ',i)
        k=[z for z in range(len(X))]
        random.shuffle(k)
        offset=0
        g=0
        for j in k:
            #print('Item: ',offset)
            x=X[j]
            y=Y[j]
            a=feedForward(x,wn,bn,y,functions,lossFunction[0],functArguments[0])
            g+=numpy.array(backProp(wn,bn,a,y,functionDerivatives,lossFunction[1],functArguments[1]))
            if (offset+1)%batchSize==0:
                wn,bn=updateWeights(wn,bn,g/batchSize,e,m)
                e=e/learningFactor
                g=0
            offset+=1
    return [wn,bn]
