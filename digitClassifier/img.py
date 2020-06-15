import io
from PIL import Image
import neural,numpy,pickle
import matplotlib.pyplot as plt
def process(img):
    p=[]
    fac = 0.99 / 255
    for i in img:
        p.append(((255-(i[0]+i[1]+i[2])/3))*fac+0.01)
    return numpy.array(p)

def filt(path,kernel,stride=1):
    g,size=convert(path)
    gg=neural.convolve2d(g.reshape(size[::-1]),kernel,stride)
    return (gg,size)

def convert(path):
    im = Image.open(path)
    size=(im.size[0],im.size[1])
    im=im.resize(size)
    px=list(im.getdata())
    g=process(px)
    return (g,size)
    
def draw(g,fname='test.png'):
    f=plt.figure()
    plt.imshow(g, cmap="Greys",aspect='equal')
    plt.axis('off')
    f.savefig(fname, bbox_inches='tight',pad_inches=0)
