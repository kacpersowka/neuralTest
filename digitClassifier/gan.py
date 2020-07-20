import sys,time,numpy,pickle,random
import matplotlib.pyplot as plt

def drawFigure(fig,fname='test.png'):
    """
    Function for drawing the output of a network into a png image using matplotlib
    """
    f=plt.figure()
    plt.imshow(fig, cmap="Greys",aspect='equal')
    plt.axis('off')
    f.savefig(fname, bbox_inches='tight',pad_inches=0)

with open("fives.pkl", "br") as fh: #Read dataset
    data = pickle.load(fh)
x=data

def rectLinear(xx):
    """
    The recommended activation function for ML due to its simple gradient
    """
    x=numpy.array(xx,dtype=float)
    for i in range(len(x)):
            x[i]=max(0,x[i])
    return x

def rectLinearDerivative(xx):
    """
    Returns a jacobian matrix with format:
    dy/dx=[[dy1/dx1 ... dy1/dxn]
             ...           ...
           [dyn/dx1 ... dyn/dxn]]
    where y=g(x) with g being the activation function
    In this case since there is no dependency between the elements of x it results in a diagonal matrix
    """
    x=numpy.array(xx,dtype=float)
    for i in range(len(x)):
            x[i]=int(x[i]>0)
    return numpy.diag(x)

def sigmoid(x):
    """
    Sigmoind activation function
    """
    return 1/(1 + numpy.exp(-numpy.array(x)))

def sigmoidDerivative(x):
    """
    Returns a jacobian matrix with format:
    dy/dx=[[dy1/dx1 ... dy1/dxn]
             ...           ...
           [dyn/dx1 ... dyn/dxn]]
    where y=g(x) with g being the activation function
    In this case since there is no dependency between the elements of x it results in a diagonal matrix
    """
    return numpy.diag((1/(1 + numpy.exp(-numpy.array(x))))*(1-(1/(1 + numpy.exp(-numpy.array(x))))))

def forward(xx,ww,b,act=rectLinear):
    """
    Applies linear transformation of weights on input (plus bias) followed by the activation function
    y=g(Wx+b)
    """
    x=numpy.append(numpy.array(xx,dtype=float),1.0) #add bias neuron
    w=numpy.array(ww,dtype=float)
    w=numpy.append(w,[[i] for i in b],axis=1) #add biases into weights
    h=w.dot(x) #apply weight and bias to layer
    y=numpy.array(act(h),dtype=float) #Apply activation function
    return y

def backward(xx,ww,b,actDer=rectLinearDerivative):
    """
    Provides the derivatives in the form of jacobian matrices w.r.t: the input x, the weights W and the bias b in that order
    """
    x=numpy.append(numpy.array(xx,dtype=float),1.0) #Add bias neuron
    w=numpy.array(ww,dtype=float)
    w=numpy.append(w,[[i] for i in b],axis=1) #add biases into weights
    h=w.dot(x) #apply weight and bias to layer
    dhdw=[]
    #This is a fairly wasteful solution, there is a faster approximation
    for i in range(len(h)): #Calculate jacobian for dhdw (jacobian for dhdx = weight matrix so no need to calculate)
        dhdw.append(([0 for j in xx]*i)+[j for j in xx]+([0 for j in xx]*(len(h)-i-1)))
    dydh=numpy.array(actDer(h))
    return [dydh.dot(ww),dydh.dot(dhdw),dydh.dot(numpy.diag([1 for i in b]))]

def train(x,wd,bd,wg,bg,e,m,k,n):
    """
    Train the networks with the following parameters and hyperparameters:
    x=dataset
    wd,bd = initial weights and biases for discriminator
    wg,bg = initial weights and biases for generator
    e = learning rate
    m = momentum
    k = amount of training cycles for discriminator per item
    n = number of epochs to train for
    """
    #initialise velocity at 0
    vdw=[0*i for i in wd]
    vdb=[0*i for i in bd]
    vgw=[0*i for i in wg]
    vgb=[0*i for i in bg]
    #Activation functions for each layer in each net (plus derivatives for backprop
    generatorActivations=[rectLinear for i in range(len(wg))]
    generatorActivationsDer=[rectLinearDerivative for i in range(len(wg))]
    discriminatorActivations=[rectLinear for i in range(len(wd)-1)]+[sigmoid]
    discriminatorActivationsDer=[rectLinearDerivative for i in range(len(wd)-1)]+[sigmoidDerivative]
    for z in range(n):
        print('Epoch: ',z)
        for i in range(len(x)):
            print('Item: ',i)
            #***Train discriminator**
            print('***Discriminator turn***')
            for K in range(k):
                #**Generate sample using generator network**
                h=numpy.random.random(len(wg[0][0])) #Generator takes random numbers as an input, can be interpreted as a prior
                for l in range(len(wg)):
                    h=forward(h,wg[l],bg[l],generatorActivations[l])
                gs=h #set generated sample variable
                #**Feedforward true sample through discriminator**
                h=x[i] #Set h to true sample
                dta=[h] #Discriminator (true) activations
                for l in range(len(wd)):
                    h=forward(h,wd[l],bd[l],discriminatorActivations[l])
                    dta.append(numpy.array(h))
                #**Feedforward generated sample through discriminator**
                h=gs #Set h to generated sample
                dga=[h] #Discriminator (generated) activations
                for l in range(len(wd)):
                    h=forward(h,wd[l],bd[l],discriminatorActivations[l])
                    dga.append(numpy.array(h))
                yt=dta[-1] #True label (should be 1)
                yg=dga[-1] #Generated label (should be 0)
                #Log operations have a negligible amount added to them for numerical stability, to prevent division by zero
                Ld=-(numpy.log(yt+1e-8)+numpy.log(1-yg+1e-8)) #Loss function for discriminator
                print('Discriminator score for true sample: ',yt)
                print('Discriminator score for generated sample: ',yg)
                print('Discriminator loss: ',Ld)
                #**Get gradient of discriminator**
                dLddyt=-1/(yt+1e-8) #Derivative of loss function w.r.t yt
                dLddyg=1/(1-yg+1e-8) #Derivative of loss function w.r.t yg
                dytdhnt,dytdwn,dytdbn=backward(dta[-2],wd[-1],bd[-1],discriminatorActivationsDer[-1])
                dygdhng,dygdwn,dygdbn=backward(dga[-2],wd[-1],bd[-1],discriminatorActivationsDer[-1])
                wdg=numpy.dot(dLddyt,dytdwn)+numpy.dot(dLddyg,dygdwn) #Weight gradients
                bdg=numpy.dot(dLddyt,dytdbn)+numpy.dot(dLddyg,dygdbn) #Bias gradients
                nvw=e*wdg.reshape(wd[-1].shape)
                nvb=e*bdg.reshape(bd[-1].shape)
                wd[-1]=wd[-1]-nvw+m*vdw[-1]
                bd[-1]=bd[-1]-nvb+m*vdb[-1]
                vdw[-1]=nvw
                vdb[-1]=nvb
                dLddhnt=numpy.dot(dLddyt,dytdhnt) #Manually compute the gradient of the loss w.r.t true label
                dLddhng=numpy.dot(dLddyg,dygdhng) #Manually compute the gradient of the loss w.r.t generated label
                ngt=dLddhnt #Neuron gradients (true)
                ngg=dLddhng #Neuron gradients (generated)
                for j in range(2,len(dta)): #Backpropagate through the remaining activations
                    #Get the neuron, weight and bias gradients for both true and generated
                    dht,dwt,dbt=backward(dta[-j-1],wd[-j],bd[-j],discriminatorActivationsDer[-j])
                    dhg,dwg,dbg=backward(dga[-j-1],wd[-j],bd[-j],discriminatorActivationsDer[-j])
                    wdg=numpy.dot(ngt,dwt)+numpy.dot(ngg,dwg)
                    bdg=numpy.dot(ngt,dbt)+numpy.dot(ngg,dbg)
                    nvw=e*wdg.reshape(wd[-j].shape)
                    nvb=e*bdg.reshape(bd[-j].shape)
                    wd[-j]=wd[-j]-nvw+m*vdw[-j]
                    bd[-j]=bd[-j]-nvb+m*vdb[-j]
                    vdw[-j]=nvw
                    vdb[-j]=nvb
                    ngt=numpy.dot(ngt,dht)
                    ngg=numpy.dot(ngg,dhg)
            print('***Generator turn***')
            #***Train generator**
            #**Generate sample using generator network**
            h=numpy.random.random(len(wg[0][0]))
            ga=[h]
            for l in range(len(wg)):
                h=forward(h,wg[l],bg[l],generatorActivations[l])
                ga.append(h)
            gs=h #set generated sample variable
            #**Feedforward generated sample through discriminator**
            h=gs #Set h to generated sample
            da=[h] #Discriminator activations
            for l in range(len(wd)):
                h=forward(h,wd[l],bd[l],discriminatorActivations[l])
                da.append(numpy.array(h))
            yg=da[-1]
            Lg=numpy.log(1-yg+1e-8) #Loss function for generator
            print('Discriminator score for generated sample: ',yg)
            print('Generator loss: ',Lg)
            #**Get gradient for generator**
            dLgdyg=1/(yg-1+1e-8) #Generator loss gradient
            #Manually calculate gradient between label and last layer in disciminator
            dydhdn=backward(da[-2],wd[-1],bd[-1],discriminatorActivationsDer[-1])[0]
            dLgdhdn=numpy.dot(dLgdyg,dydhdn)
            dg=dLgdhdn #Discriminator gradients, bridges gap from loss to end of generator
            for j in range(2,len(da)): #Backpropagate through the discriminator (neurons only since we arent updating it)
                dh=backward(da[-j-1],wd[-j],bd[-j],discriminatorActivationsDer[-j])[0]
                dg=numpy.dot(dg,dh)
            ngg=dg #Neuron generator gradients
            dgsdhn,dgsdwn,dgsdbn=backward(ga[-2],wg[-1],bg[-1],generatorActivationsDer[-1])
            wgg=numpy.dot(ngg,dgsdwn) #Weight generator gradients
            bgg=numpy.dot(ngg,dgsdbn) #Bias generator gradients
            nvw=e*wgg.reshape(wg[-1].shape)
            nvb=e*bgg.reshape(bg[-1].shape)
            wg[-1]=wg[-1]-nvw+m*vgw[-1]
            bg[-1]=bg[-1]-nvb+m*vgb[-1]
            vgw[-1]=nvw
            vgb[-1]=nvb
            ngg=numpy.dot(ngg,dgsdhn)
            for j in range(2,len(ga)): #Backpropagate through the remaining activations in the generator
                dgn,dgw,dgb=backward(ga[-j-1],wg[-j],bg[-j],generatorActivationsDer[-j])
                wgg=numpy.dot(ngg,dgw)
                bgg=numpy.dot(ngg,dgb)
                nvw=e*wgg.reshape(wg[-j].shape)
                nvb=e*bgg.reshape(bg[-j].shape)
                wg[-j]=wg[-j]-nvw+m*vgw[-j]
                bg[-j]=bg[-j]-nvb+m*vgb[-j]
                vgw[-j]=nvw
                vgb[-j]=nvb
                ngg=numpy.dot(ngg,dgn)
        #Save a snapshot of the weights at each epoch
        with open("trained"+str(n)+".pkl", "bw") as fh:
            data = (wd,bd,wg,bg)
            pickle.dump(data, fh)
    return [wd,bd,wg,bg]

if __name__=="__main__":

    if len(sys.argv)>1:
        n=int(sys.argv[1])
    else:
        n=10
    if len(sys.argv)>2:
        e=float(sys.argv[2])
    else:
        e=0.1
    if len(sys.argv)>3:
        m=float(sys.argv[3])
    else:
        m=0.8   
    if len(sys.argv)>4:
        k=int(sys.argv[4])
    else:
        k=1

    random.seed(time.time())
    ld=[28**2,64,48,32,1] #Discriminator layout
    lg=[16,32,28**2] #Generator layout
    #He initialization
    wd=[(numpy.random.random(size=[ld[i],ld[i-1]])*2-1)*(2/(ld[i-1]))**0.5 for i in range(1,len(ld))]
    bd=[numpy.array([0 for j in range(ld[i])]) for i in range(1,len(ld))]
    wg=[(numpy.random.random(size=[lg[i],lg[i-1]])*2-1)*(2/(lg[i-1]))**0.5 for i in range(1,len(lg))]
    bg=[numpy.array([0 for j in range(lg[i])]) for i in range(1,len(lg))]
    wd,bd,wg,bg=train(x,wd,bd,wg,bg,e,m,k,n)
    with open("trained.pkl", "bw") as fh:
        data = (wd,bd,wg,bg)
        pickle.dump(data, fh)
