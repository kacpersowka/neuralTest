import numpy

image_size = 28
no_of_different_labels = 10
image_pixels = image_size * image_size
data_path = "./"
train_data = numpy.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = numpy.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 
fac = 0.99 / 255
train_imgs = numpy.asfarray(train_data[:, 1:]) * fac + 0.01 #Normalise
test_imgs = numpy.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = numpy.asfarray(train_data[:, :1])
test_labels = numpy.asfarray(test_data[:, :1])

lr = numpy.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(numpy.float)
test_labels_one_hot = (lr==test_labels).astype(numpy.float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

import pickle

with open("pickled_mnist.pkl", "bw") as fh:
    data = (train_imgs, 
            test_imgs, 
            train_labels,
            test_labels,
            train_labels_one_hot,
            test_labels_one_hot)
    pickle.dump(data, fh)