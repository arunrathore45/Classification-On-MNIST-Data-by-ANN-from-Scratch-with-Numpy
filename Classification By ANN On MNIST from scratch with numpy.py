import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/temp/mnist/",one_hot=True)
import matplotlib.pyplot as plt



w0=np.random.normal(0.0,0.1,(784,500))
w1=np.random.normal(0.0,0.1,(500,10)) 
def sigmoid(x):
	return 1/(1+np.exp(-x))
 
def softmax(x):
 	k=(np.exp(x)-np.max(x))
 	return k/np.sum(k)
def local(features):
 	l0=features
 	l1=sigmoid(np.matmul(l0,w0))
 	l2=sigmoid(np.matmul(l1,w1))
 	return l2


def train(features,targets,alpha):
	global w0
	global w1
	l0=features
	l1=sigmoid(np.matmul(l0,w0))
	l2=sigmoid(np.matmul(l1,w1))

	l2_error=l2-targets
	l2_delta=l2_error
	l1_error=np.dot(l2_delta,w1.T)
	l1_delta=l1_error*(l1*(1-l1))
	w1-=alpha*(l1.T.dot(l2_delta))
	w0-=alpha*(l0.T.dot(l1_delta))
	'''global t
	global l1_m
	global l1_v
	global l2_m
	global l2_v
	l0=features
	l1=sigmoid(np.matmul(l0,w0))
	l2=sigmoid(np.matmul(l1,w1))

	l2_error=l2-targets
	l2_delta=l2_error*sigmoid(l2)
	l1_error=np.dot(l2_delta,w1.T)
	l1_delta=l1_error*sigmoid(l1)

	g1=np.matmul(l1.T,l2_delta)
	g0=np.matmul(l0.T,l1_delta)

	t+=1

	l2_m=l2_m*b1+(1-b1)*g1
	l1_m=l1_m*b1+(1-b1)*g0

	l2_v=l2_v*b2+(1-b2)*(g1*g1)
	l1_v=l1_v*b2+(1-b2)*(g0*g0)

	l2_mc=l2_m/(1-(b1*t))
	l1_mc=l1_m/(1-(b1*t))

	l2_vc=l2_v/(1-(b2*t))
	l1_vc=l1_v/(1-(b2*t))

	change_w1=l2_mc/(np.sqrt(l2_vc)+eps)
	change_w0=l1_mc/(np.sqrt(l1_vc)+eps)

	w1=w1-(alpha*change_w1)
	w0=w0-(alpha*change_w0)'''

	

def loss(y,Y):
	return np.mean((y-Y)**2)
losses=[]
for i in range(1000):
	x,y=mnist.train.next_batch(128)
	train(x,y,0.01)
	losse=loss(local(x),y)
	if i%100==0:
		print("Epoch=",i,"Loss:",losse)
	losses.append(losse)
plt.plot(losses)
plt.show()

test_data = mnist.test.images[:1000]
test_label = mnist.test.labels[:1000]

correct=0
model=local(test_data)

for i in range(model.shape[0]):
	k1=np.where(model[i]==np.max(model[i]))
	k2=np.where(test_label[i]==1)
	if k1==k2:
		correct+=1


print("Accuracy:",100*(correct/1000),"%")	
