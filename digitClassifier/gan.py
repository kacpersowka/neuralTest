import sys,time,numpy,pickle,random
import matplotlib.pyplot as plt
from data import drawFigure

with open("fives.pkl", "br") as fh:
    data = pickle.load(fh)

#train_imgs = data[0]
#test_imgs = data[1]
#train_labels = data[2]
#test_labels = data[3]
#train_labels_one_hot = data[4]
#test_labels_one_hot = data[5]

x=data

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

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

def sigmoid(x):
    return 1/(1 + numpy.exp(-numpy.array(x)))

def sigmoidDerivative(x):
    return numpy.diag((1/(1 + numpy.exp(-numpy.array(x))))*(1-(1/(1 + numpy.exp(-numpy.array(x))))))

def function1(xx,ww,b,act=rectLinear):
    x=numpy.append(numpy.array(xx,dtype=float),1.0) #add bias neuron
    w=numpy.array(ww,dtype=float)
    w=numpy.append(w,[[i] for i in b],axis=1) #add biases into weights
    h=w.dot(x) #apply weight and bias to layer
    h=numpy.array(act(h),dtype=float) #Apply activation function
    return h

def function1Derivative(xx,ww,b,actDer=rectLinearDerivative):
    x=numpy.append(numpy.array(xx,dtype=float),1.0) #Add bias neuron
    w=numpy.array(ww,dtype=float)
    w=numpy.append(w,[[i] for i in b],axis=1) #add biases into weights
    h=w.dot(x) #apply weight and bias to layer
    dhdw=[]
    for i in range(len(h)): #Calculate jacobian for dhdw (jacobian for dhdx = weight matrix so no need to calculate)
        dhdw.append(([0 for j in xx]*i)+[j for j in xx]+([0 for j in xx]*(len(h)-i-1)))
    dydh=numpy.array(actDer(h))
    return [dydh.dot(ww),dydh.dot(dhdw),dydh.dot(numpy.diag([1 for i in b]))]

def cycle(x,wd,bd,wg,bg,e,m,k,n):
    vd,vg=(0,0)
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
                h=numpy.random.random(len(wg[0][0]))
                for l in range(len(wg)):
                    h=function1(h,wg[l],bg[l],generatorActivations[l])
                gs=h #set generated sample variable
                #**Feedforward true sample through discriminator**
                h=x[i] #Set h to true sample
                dta=[h] #Discriminator (true) activations
                for l in range(len(wd)):
                    h=function1(h,wd[l],bd[l],discriminatorActivations[l])
                    dta.append(numpy.array(h))
                #**Feedforward generated sample through discriminator**
                h=gs #Set h to generated sample
                dga=[h] #Discriminator (generated) activations
                for l in range(len(wd)):
                    h=function1(h,wd[l],bd[l],discriminatorActivations[l])
                    dga.append(numpy.array(h))
                yt=dta[-1] #True label (should be 1)
                yg=dga[-1] #Generated label (should be 0)
                Ld=-(numpy.log(yt)+numpy.log(1-yg)) #Loss function for discriminator
                print('Discriminator score for true sample: ',yt)
                print('Discriminator score for generated sample: ',yg)
                print('Discriminator loss: ',Ld)
                #**Get gradient of discriminator**
                dLddyt=-1/yt #Derivative of loss function w.r.t yt
                dLddyg=1/(1-yg) #Derivative of loss function w.r.t yg
                dytdhnt,dytdwn,dytdbn=function1Derivative(dta[-2],wd[-1],bd[-1],discriminatorActivationsDer[-1])
                dygdhng,dygdwn,dygdbn=function1Derivative(dga[-2],wd[-1],bd[-1],discriminatorActivationsDer[-1])
                wdg=[numpy.dot(dLddyt,dytdwn)+numpy.dot(dLddyg,dygdwn)] #Weight gradients
                bdg=[numpy.dot(dLddyt,dytdbn)+numpy.dot(dLddyg,dygdbn)] #Bias gradients
                dLddhnt=numpy.dot(dLddyt,dytdhnt) #Manually compute the gradient of the loss w.r.t true label
                dLddhng=numpy.dot(dLddyg,dygdhng) #Manually compute the gradient of the loss w.r.t generated label
                ngt=[dLddhnt] #Neuron gradients (true)
                ngg=[dLddhng] #Neuron gradients (generated)
                for j in range(2,len(dta)): #Backpropagate through the remaining activations
                    #Get the neuron, weight and bias gradients for both true and generated
                    dht,dwt,dbt=function1Derivative(dta[-j-1],wd[-j],bd[-j],discriminatorActivationsDer[-j])
                    dhg,dwg,dbg=function1Derivative(dga[-j-1],wd[-j],bd[-j],discriminatorActivationsDer[-j])
                    wdg.insert(0,numpy.dot(ngt[0],dwt)+numpy.dot(ngg[0],dwg))
                    bdg.insert(0,numpy.dot(ngt[0],dbt)+numpy.dot(ngg[0],dbg))
                    ngt.insert(0,numpy.dot(ngt[0],dht))
                    ngg.insert(0,numpy.dot(ngg[0],dhg))
                #**Update weights on discriminator**
                for i in range(len(wd)):
                    #TODO: add momentum
                    wd[i]=wd[i]-e*wdg[i].reshape(wd[i].shape)
                    bd[i]=bd[i]-e*bdg[i].reshape(bd[i].shape)
            print('***Generator turn***')
            #***Train generator**
            #**Generate sample using generator network**
            h=numpy.random.random(len(wg[0][0]))
            ga=[h]
            for l in range(len(wg)):
                h=function1(h,wg[l],bg[l],generatorActivations[l])
                ga.append(h)
            gs=h #set generated sample variable
            #**Feedforward generated sample through discriminator**
            h=gs #Set h to generated sample
            da=[h] #Discriminator activations
            for l in range(len(wd)):
                h=function1(h,wd[l],bd[l],discriminatorActivations[l])
                da.append(numpy.array(h))
            yg=da[-1]
            Lg=numpy.log(1-yg) #Loss function for generator
            print('Discriminator score for generated sample: ',yg)
            print('Generator loss: ',Lg)
            #**Get gradient for generator**
            dLgdyg=1/(yg-1) #Generator loss gradient
            #Manually calculate gradient between label and last layer in disciminator
            dydhdn=function1Derivative(da[-2],wd[-1],bd[-1],discriminatorActivationsDer[-1])[0]
            dLgdhdn=numpy.dot(dLgdyg,dydhdn)
            dg=dLgdhdn #Discriminator gradients, bridges gap from loss to end of generator
            for j in range(2,len(da)): #Backpropagate through the discriminator (neurons only since we arent updating it)
                dh=function1Derivative(da[-j-1],wd[-j],bd[-j],discriminatorActivationsDer[-j])[0]
                dg=numpy.dot(dg,dh)
            ngg=[dg] #Neuron generator gradients
            dgsdhn,dgsdwn,dgsdbn=function1Derivative(ga[-2],wg[-1],bg[-1],generatorActivationsDer[-1])
            wgg=[numpy.dot(ngg[0],dgsdwn)] #Weight generator gradients
            bgg=[numpy.dot(ngg[0],dgsdbn)] #Bias generator gradients
            ngg.insert(0,numpy.dot(ngg[0],dgsdhn))
            for j in range(2,len(ga)): #Backpropagate through the remaining activations in the generator
                dgn,dgw,dgb=function1Derivative(ga[-j-1],wg[-j],bg[-j],generatorActivationsDer[-j])
                wgg.insert(0,numpy.dot(ngg[0],dgw))
                bgg.insert(0,numpy.dot(ngg[0],dgb))
                ngg.insert(0,numpy.dot(ngg[0],dgn))
            #**Update weights on generator**
            for i in range(len(wd)):
                #TODO: add momentum
                wg[i]=wg[i]-e*wgg[i].reshape(wg[i].shape)
                bg[i]=bg[i]-e*bgg[i].reshape(bg[i].shape)
    return [wd,bd,wg,bg]

if __name__=="__main__":

    if len(sys.argv)>1:
        n=int(sys.argv[1])
    else:
        n=1
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
    ld=[28**2,32,32,1] #Discriminator layout
    lg=[64,48,32,28**2] #Generator layout
    #He initialization
    wd=[(numpy.random.random(size=[ld[i],ld[i-1]])*2-1)*(2/(ld[i-1]))**0.5 for i in range(1,len(ld))]
    bd=[numpy.array([0 for j in range(ld[i])]) for i in range(1,len(ld))]
    wg=[(numpy.random.random(size=[lg[i],lg[i-1]])*2-1)*(2/(lg[i-1]))**0.5 for i in range(1,len(lg))]
    bg=[numpy.array([0 for j in range(lg[i])]) for i in range(1,len(lg))]
    wd,bd,wg,bg=cycle(x,wd,bd,wg,bg,e,m,k,n)
    with open("trained.pkl", "bw") as fh:
        data = (wd,bd,wg,bg)
        pickle.dump(data, fh)
