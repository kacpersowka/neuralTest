from img import *
import sys

def gaussian(file,stride):
    return filt(file,[[1,2,1],[2,3,2],[1,2,1]],stride)[0]

def inverse(file,stride):
    return filt(file,[[0,0,0],[0,-1,0],[0,0,0]],stride)[0]

def sobelX(file,stride):
    return filt(file,[[-1,0,1],[-2,0,2],[-1,0,1]],stride)[0]

def sobelY(file,stride):
    return filt(file,[[-1,-2,-1],[0,0,0],[1,2,1]],stride)[0]

def sobel(file,stride):
    x=sobelX(file,stride)
    y=sobelY(file,stride)
    xy=[]
    for i in range(len(x)):
        xy.append([])
        for j in range(len(x[i])):
            xy[i].append((x[i][j]**2+y[i][j]**2)**0.5)
    return xy
file=sys.argv[1]
kernel=sys.argv[2]
n=1
if len(sys.argv)>3:
    n=int(sys.argv[3])
a=eval(kernel+'("'+file+'",'+str(n)+')')
draw(a,file+'_'+kernel+'_'+str(n)+'.png')