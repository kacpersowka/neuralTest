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

def maxPool(inp,size=2):
    out=[]
    inp=numpy.array(inp)
    for i in range(0,len(inp),size):
        for j in range(0,len(inp[0]),size):
            out.append(max(inp[i:size+i][:,j:size+j].flatten()))
    return numpy.array(out).reshape((proper_round((len(inp))/size),len(out)//proper_round((len(inp))/size)))

def maxPoolDerivative(inp,size=2):
    inp=numpy.array(inp)
    out=[]
    for i in range(0,len(inp),size):
        for j in range(0,len(inp[0]),size):
            a=inp[i:size+i][:,j:size+j]
            b=numpy.where(a==max(a.flatten()))
            out.append([0 for k in inp.flatten()])
            out[-1][(i+b[0][0])*(len(inp)-1)+(j+b[1][0])]=1
    return out

def convolve2d(inp,kernel,stride=1,mode='same'):
    kernel=numpy.array(kernel)
    if mode=='full':
        inp=numpy.pad(inp,len(kernel)-1)
        padOffset=(len(kernel)//2)+(len(kernel)-1)//2
    elif mode=='same':
        inp=numpy.pad(inp,(len(kernel)-1)//2)
        padOffset=2*((len(kernel)-1)//2)
    elif mode=='valid':
        inp=numpy.array(inp)
        padOffset=(len(kernel)//2)+(len(kernel)-1)//2
    s=[]
    for i in range(0,(len(inp)-padOffset),stride):
        for j in range(0,(len(inp[0])-padOffset),stride):
            s.append(sum(numpy.multiply(inp[i:len(kernel)+i][:,j:len(kernel)+j],kernel[::-1][:,::-1]).flatten()))
    return numpy.array(s).reshape((proper_round((len(inp)-padOffset)/stride),len(s)//proper_round((len(inp)-padOffset)/stride)))

def convolve2dDerivative(inp,kernel,stride=1,mode='same'):
    kernel=numpy.array(kernel)
    if mode=='full':
        inp=numpy.pad(inp,len(kernel)-1)
        pad=len(kernel)-1
        padOffset=(len(kernel)//2)+(len(kernel)-1)//2
    elif mode=='same':
        inp=numpy.pad(inp,(len(kernel)-1)//2)
        pad=(len(kernel)-1)//2
        padOffset=2*((len(kernel)-1)//2)
    elif mode=='valid':
        inp=numpy.array(inp)
        pad=0
        padOffset=(len(kernel)//2)+(len(kernel)-1)//2
    inpDer=[] #dcdx
    kerDer=[] #dcdk
    for i in range(0,(len(inp)-padOffset),stride):
        for j in range(0,(len(inp[0])-padOffset),stride):
            temp=numpy.zeros(inp.shape)
            temp[i:len(kernel)+i][:,j:len(kernel)+j]=kernel[::-1][:,::-1]
            if pad!=0:
                inpDer.append(temp[pad:-pad][:,pad:-pad].flatten())
            else:
                inpDer.append(temp.flatten())
            kerDer.append(inp[i:len(kernel)+i][:,j:len(kernel)+j][::-1][:,::-1].flatten())
    return [numpy.array(inpDer),numpy.array(kerDer)]

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

def function2(xx,kernel,b,act=expRectLinear,pool=maxPool,poolArgs=[2],stride=1,mode='same'):
    x=numpy.array(xx,dtype=float)
    c=convolve2d(x,kernel,stride,mode)+b
    h=act(c.flatten()).reshape(c.shape)
    y=pool(h,*poolArgs)
    return y

def function2Derivative(xx,kernel,b,act=expRectLinear,actDer=expRectLinearDerivative,poolDer=maxPoolDerivative,poolDerArgs=[2],stride=1,mode='same'):
    x=numpy.array(xx,dtype=float)
    c=convolve2d(x,kernel,stride,mode)+b
    h=act(c.flatten()).reshape(c.shape)
    dydh=poolDer(h,*poolDerArgs)
    dhdc=actDer(c.flatten())
    dcdx,dcdk=convolve2dDerivative(x,kernel,stride,mode)
    dydc=numpy.dot(dydh,dhdc)
    dydx=numpy.dot(dydc,dcdx)
    dydk=numpy.dot(dydc,dcdk)
    dhdb=numpy.sum(dhdc)
    dydb=numpy.dot(dydh,dhdb)
    return [dydx,dydk,dydb]

def mseGradient(x,y): #dL/dX (w.r.t predicted)
    return [[2*(x[i]-y[i])/len(x) for i in range(len(x))]]

def backProp(w,b,a,y,functionDer=function1Derivative,lossDerivative=mseGradient,functArguments=[[rectLinearDerivative],[lambda x:[1 for i in x]]]):
    if type(functionDer)!=list:
        functionDer=[functionDer for i in range(len(w))]
    if len(functArguments)==2 and len(w)>2:
        functArguments=functArguments[:-1]+[functArguments[0] for i in range(len(functArguments),len(w))]+functArguments[-1:]
    neuronGradients=[[] for i in a[:-1]]
    weightGradients=[[] for i in w]
    neuronGradients.append(numpy.array([1.0])) #manually calculate loss gradient
    neuronGradients[-2]=numpy.array(lossDerivative(a[-2],y)) #manually calculate prediction gradient
    #grad_table[-2]=numpy.array(grad_table[-2][0])
    for i in range(len(a)-3,-1,-1): #skip loss and prediction
        der=(functionDer[i](a[i],w[i],b[i],*functArguments[i]))
        neuronGradients[i]=numpy.dot(neuronGradients[i+1],der[0])
        weightGradients[i]=numpy.multiply(neuronGradients[i+1].transpose(),der[1])
    return [neuronGradients,weightGradients]

def generateRandomWeightsAndBiases(l,wf=1,bf=0,weightMap=None,biasMap=None):
    w=[]
    b=[]
    for i in range(1,len(l)):
        if weightMap==None or weightMap[i]==None or weightMap[i]==[]:
            #w.append([[random.random()*((2/l[i-1])**0.5)*wf for k in range(l[i-1])] for j in range(l[i])])
            w.append((numpy.random.random(size=[l[i],l[i-1]])*2-1)*(2/(l[i-1]))**0.5*wf)
        else:
            w.append(weightMap[i])
        if biasMap==None or biasMap[i]==None or biasMap[i]==[]:
            b.append([(random.random()*2-1)*bf for k in range(l[i])])
        else:
            b.append(biasMap[i])
    return [w,b]

def generateRandomKernelsAndBiases(l,wf=1,bf=0,weightMap=None,biasMap=None):
    w=[]
    b=[]
    for i in range(len(l)):
        if weightMap==None or weightMap[i]==None or weightMap[i]==[]:
            #w.append([[random.random()*wf for k in range(l[i])] for j in range(l[i])])
            w.append((numpy.random.random(size=[l[i],l[i]])*2-1)*(2/(l[i-1]))**0.5*wf)
        else:
            w.append(weightMap[i])
        if biasMap==None or biasMap[i]==None or biasMap[i]==[]:
            b.append((random.random()*2-1)*bf)
        else:
            b.append(biasMap[i])
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

def feedForwardCNN(inp,kernels,cnnBiases,mlpW,mlpB,y,functions=function2,mlpFunctions=function1,lossFunction=crossEntropy,mlpFunctArguments=[[expRectLinear],[softmax]],functArguments=[[expRectLinear,maxPool,[2],1,'same']]):
    if type(functions)!=list:
        functions=[functions for i in range(len(kernels))]
    if len(functArguments)==1:
        functArguments=[functArguments[0] for i in range(len(kernels))]
    x=inp
    a=[inp]
    for i in range(len(kernels)): #Feed forward CNN layers
        x=functions[i](x,kernels[i],cnnBiases[i],*functArguments[i])
        #x=pools[i](cnnAct[i]((convolve2d(x,kernels[i])+cnnBiases[i]).flatten()).reshape(x.shape),*poolArguments[i])
        a.append(x)
    if mlpW!=None:
        a+=feedForward(x.flatten(),mlpW,mlpB,y,mlpFunctions,lossFunction,mlpFunctArguments)[1:]
    return a

def backPropCNN(kernels,cnnBiases,mlpW,mlpB,a,y,functionDer=function2Derivative,mlpFunctionDer=function1Derivative,lossDerivative=crossEntropyGradient,mlpFunctArguments=[[rectLinearDerivative],[softmaxDerivative]],functArguments=[[expRectLinear,expRectLinearDerivative,maxPoolDerivative,[2],1,'same']]):
    if type(functionDer)!=list:
        functionDer=[functionDer for i in range(len(kernels))]
    if len(functArguments)==1:
        functArguments=[functArguments[0] for i in range(len(kernels))]
    tempSh=a[len(kernels)].shape
    a[len(kernels)]=numpy.append(a[len(kernels)].flatten(),1)
    neuronGradients,weightGradients=backProp(mlpW,mlpB,a[len(kernels):],y,mlpFunctionDer,lossDerivative,mlpFunctArguments)
    biasGradients=[]
    a[len(kernels)]=numpy.delete(a[len(kernels)],-1).reshape(tempSh)
    for i in range(len(a)-len(neuronGradients)-1,-1,-1):
        der=functionDer[i](a[i],kernels[i],cnnBiases[i],*functArguments[i])
        biasGradients.insert(0,numpy.dot(neuronGradients[0],der[2]))
        weightGradients.insert(0,numpy.dot(neuronGradients[0],der[1]))
        neuronGradients.insert(0,numpy.dot(neuronGradients[0],der[0]))
    return [neuronGradients,weightGradients,biasGradients]

def updateKernels(kernels,b,g,e,m,v):
    newKernels=[]
    newBiases=[]
    for i in range(len(kernels)):
        if v[0]!=0:
            newV=e*g[1][i].reshape(kernels[i].shape)
            newKernels.append(kernels[i]-newV+m*v[0][i].reshape(kernels[i].shape))
            v[0][i]=newV.flatten()
            newV=numpy.sum(e*g[2][i])
            newBiases.append(b[i]-newV+m*v[1][i])
            v[1][i]=newV
        else:
            newKernels.append(kernels[i]-e*g[1][i].reshape(kernels[i].shape))
            newBiases.append(b[i]-numpy.sum(e*g[2][i]))
    return [newKernels,newBiases,v]

def updateWeights(w,b,g,e,m,v):
    wb=[]
    for i in range(len(w)): #merge biases into weights
        if (type(b[i])==list or type(b[i])==type(numpy.array(0))):
            wb.append(numpy.append(w[i],[[j] for j in b[i]],axis=1))
        else:
            wb.append(numpy.append(w[i],b[i]))
    for i in range(len(wb)):
        for j in range(len(wb[i])):
            if v!=0:
                newV=e*g[i][j]
                wb[i][j]=wb[i][j]-newV+m*v[i][j]
                v[i][j]=newV
            else:
                wb[i][j]=wb[i][j]-e*g[i][j]
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
    return [wn,bn,v]

def train(X,Y,l,n,e,m,init=generateRandomWeightsAndBiases,initArgs=[1,0],functions=function1,lossFunction=[mse,mseGradient],functionDerivatives=function1Derivative,functArguments=[[[rectLinear],[lambda x:x]],[[rectLinearDerivative],[lambda x:numpy.diag([1 for i in x])]]]):
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
            g=backProp(wn,bn,a,y,functionDerivatives,lossFunction[1],functArguments[1])[1]
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
            g+=numpy.array(backProp(wn,bn,a,y,functionDerivatives,lossFunction[1],functArguments[1])[1])
            if (offset+1)%batchSize==0:
                wn,bn=updateWeights(wn,bn,g/batchSize,e,m)
                e=e/learningFactor
                g=0
            offset+=1
    return [wn,bn]
