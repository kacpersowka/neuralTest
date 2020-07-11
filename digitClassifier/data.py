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
w,b=generateRandomWeightsAndBiases([16,16,10],5,0)
#yy=[0,0,0,0,0,1,0,0,0,0]
xx=numpy.array(
    [[[0,1,2,3,4],[0,0,1,2,3],[0,0,0,1,2],[0,0,0,0,1],[0,0,0,0,0]],
    [[4,3,2,1,0],[3,2,1,0,0],[2,1,0,0,0],[1,0,0,0,0],[0,0,0,0,0]],
    [[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,2],[0,0,1,2,3],[0,1,2,3,4]],
    [[0,0,0,0,0],[1,0,0,0,0],[2,1,0,0,0],[3,2,1,0,0],[4,3,2,1,0]]])+1
X=[numpy.multiply(numpy.random.random((5,5)),xx[i%4]) for i in range(40)]
yy=numpy.array([[[0.01,0.97],[0.01,0.01]],[[0.97,0.01],[0.01,0.01]],[[0.01,0.01],[0.01,0.97]],[[0.01,0.01],[0.97,0.01]]])

def test(X,yy,k,b,e,n):
    for j in range(n):
        for i in range(len(X)):
            h=function2(X[i],k,b,mode='valid')
            l=crossEntropy(h.flatten(),yy[i%4].flatten())
            dldh=crossEntropyGradient(h.flatten(),yy[i%4].flatten())
            dhdX,dhdk,dhdb=function2Derivative(X[i],k,b,mode='valid')
            dldX=numpy.dot(dldh,dhdX)
            dldk=numpy.dot(dldh,dhdk)
            dldb=numpy.dot(dldh,dhdb)
            k,b=updateKernels([k],[b],[[dldX],[dldk],[dldb]],e,0)
            print(b)
            k,b=[k[0],b[0]]
    return [k,b]
    
#kernels=numpy.array([[[1,2,1],[2,3,2],[1,2,1]],[[-1,-2,-1],[0,0,0],[1,2,1]],[[0,0,0],[0,-1,0],[0,0,0]]])
#biases=[0,0,0]
kernels,biases=generateRandomKernelsAndBiases([3,3,3],1,0)
kernels=numpy.array(kernels)

def cycle(x,y,w,b,kernels,biases,e):
    for i in range(len(x)):
        yb=[0.0 for j in range(10)]
        yb[int(y[i])]=1.0
        a=feedForwardCNN(x[i].reshape(28,28),kernels,biases,w,b,yb)
        print('Outputs: ',a[-2])
        print('Expected: ',yb)
        print('Error: ',a[-1])
        ng,wg,bg=backPropCNN(kernels,biases,w,b,a,yb)
        w,b=updateWeights(w,b,wg[len(kernels):],e)
        kernels,biases=updateKernels(kernels,biases,[ng[:len(kernels)+1],wg[:len(kernels)],bg],e)
    return [w,b,kernels,biases]

def drawFigure(fig,fname='test.png'):
    f=plt.figure()
    plt.imshow(fig, cmap="Greys")
    f.savefig(fname)
    
if __name__=='__main__':
    e=0.1
    nn=1
    sx,sy=[x,y]
    n=len(sx)
    sx,sy=[sx[:n],sy[:n]]
    for i in range(nn):
        print('EPOCH: ',i)
        w,b,kernels,biases=cycle(sx,sy,w,b,kernels,biases,e)
    with open("trainedCNN.pkl", "bw") as fh:
        data = (w,b,kernels,biases)
        pickle.dump(data, fh)