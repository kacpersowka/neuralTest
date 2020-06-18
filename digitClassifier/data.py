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

def drawFigure(fig,fname='test.png'):
    f=plt.figure()
    plt.imshow(fig, cmap="Greys")
    f.savefig(fname)