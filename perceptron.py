import numpy as np
import matplotlib.pyplot as plt

def draw(x1,x2):
    ln=plt.plot(x1,x2)
    return ln
    

def sigmoid(score):
    return 1/(1+np.exp(-score))

def calculate_error(line_param,pts,y):
    m = pts.shape[0]
    p = sigmoid(pts*line_param)
    cross_entropy = -(1/m)*(np.log(p).T * y + np.log(1-p).T*(1-y)) 
    return cross_entropy

def gradient(line_param,pts,y,alpha):
    for i in range(1000):
        m = pts.shape[0]
        p = sigmoid(pts*line_param)
        gradient = (pts.T * (p-y))*(alpha/m)
        line_param = line_param - gradient
        w1 = line_param.item(0)
        w2 = line_param.item(1)
        b = line_param.item(2)
        x1 = np.array([pts[:,0].min(),pts[:,0].max()])
        x2 = -b / w2 + x1 * (-w1 / w2)
    draw(x1,x2)

pts=100
np.random.seed(0)
bias=np.ones(pts)
top_region=np.array([np.random.normal(10,2,pts),np.random.normal(12,2,pts),bias]).T
bottom_region=np.array([np.random.normal(6,2,pts),np.random.normal(8,2,pts),bias]).T

all_pts=np.vstack((top_region,bottom_region))
line_param = np.matrix([np.zeros(3)]).T
y = np.array([np.zeros(pts),np.ones(pts)]).reshape(pts*2,1)


_,ax=plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0],top_region[:,1],color="red")
ax.scatter(bottom_region[:,0],bottom_region[:,1],color="blue")
gradient(line_param,all_pts,y,0.06)
plt.show()
