import numpy
import random

def rectLinear(xx):
    x=numpy.array(xx)
    for i in range(len(x)):
            x[i]=max(0,x[i]) #apply activation function
    return x

def rectLinearDerivative(xx):
    x=numpy.array(xx)
    for i in range(len(x)):
            x[i]=int(x[i]>0) #apply activation function
    return x

def softmax(x):
    z=numpy.array(x)-max(x)
    return numpy.exp(z) / numpy.sum(numpy.exp(z), axis=0)

def softmaxDerivative(x):
    z=numpy.array(x)-max(x)
    s=numpy.sum(numpy.exp(z), axis=0)
    return (numpy.multiply(s,numpy.exp(z))-numpy.multiply(numpy.exp(z),numpy.exp(z)))/s**2

def crossEntropy(q,p):
    z=numpy.array(q)+1e-7 #For numerical stability
    return -sum(p*numpy.log(z))

def crossEntropyGradient(q,p): #w.r.t q
    z=numpy.array(q)+1e-7
    return [[p[i]/z[i] for i in range(len(p))]]

def mse(x,y):
    s=0
    for i in range(len(x)):
        s+=(x[i]-y[i])**2
    return s/len(x)

def getPredictions(w,x,act=rectLinear):
    w=numpy.array(w)
    train=numpy.array(x)
    train=train.dot(w)
    for i in range(len(train)):
        train[i]=act(train[i])
    return train

def function1(xx,ww,b,act=rectLinear):
    x=numpy.append(numpy.array(xx),1) #add bias column
    w=numpy.array(ww)
    w=numpy.append(w,[[i] for i in b],axis=1) #add biases into weights
    h=w.dot(x) #apply weight and bias to layer
    h=numpy.array(act(h))
    #for i in range(len(h)):
    #        h[i]=act(h[i]) #apply activation function
    return h

def function1Derivative(xx,ww,b,actDer=rectLinearDerivative):
    x=numpy.array(xx)
    w=numpy.array(ww)
    w=numpy.append(w,[[i] for i in b],axis=1) #add biases into weights
    h=w.dot(x) #apply weight and bias to layer
    h=numpy.array(actDer(h))
    #for i in range(len(h)):
    #        h[i]=actDer(h[i]) #apply activation function
    return [numpy.multiply(h.reshape([len(h),-1]),ww),h.reshape([len(h),-1]).dot([x])]

def mseGradient(x,y): #dL/dX (w.r.t predicted)
    return [[2*(x[i]-y[i])/len(x) for i in range(len(x))]]

def backProp(w,b,a,y,functionDer=function1Derivative,lossDerivative=mseGradient,functArguments=[[rectLinearDerivative],[lambda x:[1 for i in x]]]):
    if type(functionDer)!=list:
        functionDer=[functionDer for i in range(len(w))]
    if len(functArguments)==2 and len(w)>2:
        functArguments=functArguments[:-1]+[functArguments[0] for i in range(len(functArguments),len(w))]+functArguments[-1:]
    jacobians=[[] for i in a[:-1]]
    weightGradients=[[] for i in w]
    jacobians.append(numpy.array([1])) #manually calculate loss gradient
    jacobians[-2]=numpy.array(lossDerivative(a[-2],y)) #manually calculate prediction gradient
    #grad_table[-2]=numpy.array(grad_table[-2][0])
    for i in range(len(a)-3,-1,-1): #skip loss and prediction
        der=(functionDer[i](a[i],w[i],b[i],*functArguments[i]))
        jacobians[i]=numpy.array(numpy.dot(jacobians[i+1],der[0]))
        weightGradients[i]=numpy.multiply(jacobians[i+1].transpose(),der[1])
    return weightGradients

def generateRandomWeightsAndBiases(l):
    w=[]
    b=[]
    for i in range(1,len(l)):
        w.append([[random.random() for k in range(l[i-1])] for j in range(l[i])])
        b.append([random.random()/100 for k in range(l[i])])

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
    owb=[[0 for j in range(len(wb[i]))] for i in range(len(wb))]
    for i in range(len(wb)):
        for j in range(len(wb[i])):
            wb[i][j]=wb[i][j]-e*g[i][j]+m*owb[i][j]
            owb[i][j]=e*g[i][j]+m*owb[i][j]
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

def train(X,Y,l,n,e,m,functions=function1,lossFunction=[mse,mseGradient],functionDerivatives=function1Derivative,functArguments=[[[rectLinear],[lambda x:x]],[[rectLinearDerivative],[lambda x:[1 for i in x]]]]):
    wn,bn=generateRandomWeightsAndBiases([len(X[0])]+l)
    for i in range(n):
        #print('Pass: ',i)
        for j in range(len(X)):
            #print('Item: ',j)
            x=X[j]
            y=Y[j]
            a=feedForward(x,wn,bn,y,functions,lossFunction[0],functArguments[0])
            g=backProp(wn,bn,a,y,functionDerivatives,lossFunction[1],functArguments[1])
            wn,bn=updateWeights(wn,bn,g,e,m)
    return [wn,bn]
