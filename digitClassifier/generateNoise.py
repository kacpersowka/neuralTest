#add noise
import pickle,random,sys
x=[]
y=[]
if len(sys.argv)>1:
    na=int(sys.argv[1])
else:
    na=10000
    
if len(sys.argv)>1:
    nb=int(sys.argv[1])
else:
    nb=10000

for i in range(na):
    x.append([random.random() for j in range(28**2)])
    y.append([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.99])
    
with open("noise.pkl", "bw") as fh:
    data = (x,y)
    pickle.dump(data, fh)