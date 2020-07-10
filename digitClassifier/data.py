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
w,b=generateRandomWeightsAndBiases([16,16,10],1,0)
yy=[0,0,0,0,0,1,0,0,0,0]
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
        #ng,wg=backPropCNN(kernels,biases,w,b,a,yb)
        #w,b=updateWeights(w,b,wg[len(kernels):],e)
        #kernels,biases=updateKernels(kernels,biases,[ng[:len(kernels)+1],wg[:len(kernels)]],e)
    return [w,b,kernels,biases]

def drawFigure(fig,fname='test.png'):
    f=plt.figure()
    plt.imshow(fig, cmap="Greys")
    f.savefig(fname)
    
if __name__=='__main__':
    e=0.1
    nn=1
    sx,sy=[x,y]
    sx,sy=[sx[:n],sy[:n]]
    n=len(sx)
    for i in range(nn):
        print('EPOCH: ',i)
        w,b,kernels,biases=cycle(sx,sy,w,b,kernels,biases,e)
    with open("trainedCNN.pkl", "bw") as fh:
        data = (w,b,kernels,biases)
        pickle.dump(data, fh)