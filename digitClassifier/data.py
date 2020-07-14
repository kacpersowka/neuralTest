from neural import *
import sys,time
import pickle
import matplotlib.pyplot as plt

with open("miniMNIST.pkl", "br") as fh:
    data = pickle.load(fh)

#train_imgs = data[0]
#test_imgs = data[1]
#train_labels = data[2]
#test_labels = data[3]
#train_labels_one_hot = data[4]
#test_labels_one_hot = data[5]

x=data[0]
y=data[1]

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

random.seed(time.time())
w,b=generateRandomWeightsAndBiases([9,16,4],1,0)
#yy=[0,0,0,0,0,1,0,0,0,0]
xx=numpy.array([
    numpy.repeat(numpy.repeat([[0,1,2,4,8],[0,0,1,2,4],[0,0,0,1,2],[0,0,0,0,1],[0,0,0,0,0]],2,axis=0),2,axis=1),
    numpy.repeat(numpy.repeat([[8,4,2,1,0],[4,2,1,0,0],[2,1,0,0,0],[1,0,0,0,0],[0,0,0,0,0]],2,axis=0),2,axis=1),
    numpy.repeat(numpy.repeat([[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,2],[0,0,1,2,4],[0,1,2,4,8]],2,axis=0),2,axis=1),
    numpy.repeat(numpy.repeat([[0,0,0,0,0],[1,0,0,0,0],[2,1,0,0,0],[4,2,1,0,0],[8,4,2,1,0]],2,axis=0),2,axis=1)])+1
X=[numpy.multiply(numpy.random.random((10,10)),xx[i%4]) for i in range(10000)]
yy=numpy.array([[0.01,0.99,0.01,0.01],[0.99,0.01,0.01,0.01],[0.01,0.01,0.01,0.99],[0.01,0.01,0.99,0.01]])
#yy=numpy.array([[0.01,0.01,0.99,0.01],[0.01,0.01,0.01,0.99],[0.99,0.01,0.01,0.01],[0.01,0.99,0.01,0.01]])

"""
def test(X,yy,k,b,e,n):
    for j in range(n):
        for i in range(len(X)):
            h=function2(X[i],k,b,mode='valid')
            l=mse(h.flatten(),yy[i%4].flatten())
            dldh=mseGradient(h.flatten(),yy[i%4].flatten())
            dhdX,dhdk,dhdb=function2Derivative(X[i],k,b,mode='valid')
            dldX=numpy.dot(dldh,dhdX)
            dldk=numpy.dot(dldh,dhdk)
            dldb=numpy.dot(dldh,dhdb)
            k,b=updateKernels([k],[b],[[dldX],[dldk],[dldb]],e,0)
            print('Output: ',h.flatten())
            print('Expected: ',yy[i%4].flatten())
            print('Error: ',l)
            k,b=[k[0],b[0]]
    return [k,b]
"""

def test(X,yy,k,kb,w,b,e,n):
    for j in range(n):
        for i in range(len(X)):
            h=function2(X[i],k,kb,mode='valid')
            y=function1(h.flatten(),w,b,act=softmax)
            l=crossEntropy(y,yy[i%4])
            dldy=crossEntropyGradient(y,yy[i%4])
            dydh,dydw=function1Derivative(numpy.append(h.flatten(),1.0),w,b,actDer=softmaxDerivative)
            dhdX,dhdk,dhdb=function2Derivative(X[i],k,kb,mode='valid')
            dldh=numpy.dot(dldy,dydh)
            dldw=numpy.dot(dldy,dydw)
            dldX=numpy.dot(dldh,dhdX)
            dldk=numpy.dot(dldh,dhdk)
            dldb=numpy.dot(dldh,dhdb)
            w,b=updateWeights([w],[b],dldw,e,0)
            k,kb=updateKernels([k],[kb],[[dldX],[dldk],[dldb]],e,0)
            print('Output: ',y)
            print('Expected: ',yy[i%4])
            print('Error: ',l)
            k,kb,w,b=[k[0],kb[0],w[0],b[0]]
    return [k,kb,w,b]
    
#kernels=numpy.array([[[1,2,1],[2,3,2],[1,2,1]],[[-1,-2,-1],[0,0,0],[1,2,1]],[[0,0,0],[0,-1,0],[0,0,0]]])
#biases=[0,0,0]
kernels,biases=generateRandomKernelsAndBiases([3,3],1,0)
kernels=numpy.array(kernels)

def cycle(x,y,w,b,kernels,biases,e,m):
    v=[0,0]
    for i in range(len(x)):
        #yb=[0.0 for j in range(10)]
        #yb[int(y[i])]=1.0
        yb=y[i%4]
        #yb=y[i]
        #a=feedForwardCNN(x[i].reshape(28,28),kernels,biases,w,b,yb)
        a=feedForwardCNN(x[i],kernels,biases,w,b,yb)
        print('Outputs: ',a[-2])
        print('Expected: ',yb)
        print('Error: ',a[-1])
        ng,wg,bg=backPropCNN(kernels,biases,w,b,a,yb)
        if v[0]!=0:
            w,b,vn=updateWeights(w,b,wg[len(kernels):],e,m,v[0][len(kernels):])
            v[0]=vn
            kernels,biases,vn=updateKernels(kernels,biases,[ng[:len(kernels)+1],wg[:len(kernels)],bg],e,m,[v[0][:len(kernels)],v[1]])
            v[0]+=vn[0]
            v[1]=v[1]
        else:
            w,b,vn=updateWeights(w,b,wg[len(kernels):],e,m,v[0])
            v[0]=vn
            kernels,biases,vn=updateKernels(kernels,biases,[ng[:len(kernels)+1],wg[:len(kernels)],bg],e,m,[0,0])
            v[0]+=vn[0]
            v[1]=v[1]
    return [w,b,kernels,biases]

def drawFigure(fig,fname='test.png'):
    f=plt.figure()
    plt.imshow(fig, cmap="Greys")
    f.savefig(fname)
    
if __name__=='__main__':
    e=0.01
    nn=1
    m=0.8
    sx,sy=[X,yy]
    n=len(sx)
    sx,sy=[sx[:n],sy[:n]]
    for i in range(nn):
        print('EPOCH: ',i)
        w,b,kernels,biases=cycle(sx,sy,w,b,kernels,biases,e,m)
    with open("trainedCNN.pkl", "bw") as fh:
        data = (w,b,kernels,biases)
        pickle.dump(data, fh)