import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import imp
import matplotlib.pyplot as plt
import random

data=np.load('../data/processed_data.npz')
pids=data['pids']
X=data['X']
X_sym=data['X_sym']
Y=data['Y']
Z=data['Z']

question = np.concatenate((Y,Z),axis=1)
print(question.shape)

x_train = question[0:120,:]
x_val = question[120:145,:]
x_test = question
#print(y_val)

PLOT_COLORS = ['red', 'green', 'blue', 'orange','purple','pink']

color_array=np.zeros((len(x_train[:,0]),1))
classes=np.zeros((len(x_test[:,0]),1))
#print(np.shape(color_array))

num_cen=4
centroids_prev=np.zeros((num_cen,len(x_train[0,:])))
centroids=np.zeros((num_cen,len(x_train[0,:])))
len_cen=len(centroids[0,:])
for i in range(0,num_cen):
	centroids[i,:]=x_train[random.randint(0,len(x_train[:,0])-1),:]
#print(centroids)

mse=1e10

epsilon=1e-5
it=0

while mse>epsilon:    #100 iterations to start
	for i in range(0,len(x_train[:,0])):
		mindist=1e6
		for ii in range(0,num_cen):
			distance=(centroids[ii,:].reshape((len_cen,1))-x_train[i,:].reshape((len_cen,1))).T.dot((centroids[ii,:].reshape((len_cen,1))-x_train[i,:].reshape((len_cen,1))))[0,0]
			if distance < mindist:
				mindist=distance
				minmu=ii+1
		color_array[i]=minmu

	mu_sums=np.zeros((num_cen))
	mu_sumx=np.zeros((num_cen,len_cen))
	for i in range(0,len(x_train[:,0])):
		mu_sums[int(color_array[i]-1)]=mu_sums[int(color_array[i]-1)]+1
		mu_sumx[int(color_array[i]-1)]=mu_sumx[int(color_array[i]-1)]+x_train[i,:]
	#print(mu_sumx)
	for i in range(0,len(mu_sums)):
		if mu_sums[i]!=0:
			mu_sumx[i,:]=mu_sumx[i,:]/mu_sums[i]
	centroids_prev=centroids
	centroids=mu_sumx

	mse=np.sum(np.square(centroids_prev-centroids))
	print(mse)

	print(it)
	it=it+1

for i in range(0,len(x_test[:,0])):
	mindist=1e6
	for ii in range(0,num_cen):
		distance=(centroids[ii,:].reshape((len_cen,1))-x_test[i,:].reshape((len_cen,1))).T.dot((centroids[ii,:].reshape((len_cen,1))-x_test[i,:].reshape((len_cen,1))))[0,0]
		if distance < mindist:
			mindist=distance
			minmu=ii+1
	classes[i]=minmu

m = X.shape[0]
n = X.shape[1]

x_train=X[0:120,:,:]
x_val=X[120:145,:,:]
x_test = X[145:,:,:]

x_train_r = np.reshape(x_train,[120,n*14],order='F')
x_val_r = np.reshape(x_val,[145-120,n*14],order='F')
x_test_r = np.reshape(x_test,[m-145,n*14],order='F')

#look at first question
y_train=classes[0:120,0]
y_val = classes[120:145,0]
y_test = classes[145:]
k_y_min = int(np.min(classes))
k_y = int(np.max(classes))

log = LogisticRegression(C = 1e-4, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000)
log.fit(x_train_r,y_train)

pred_train = log.predict(x_train_r)
pred_val = log.predict(x_val_r)

#print(y_val)
#print(pred_val)

#custom accuracy
unweight_train_acc = np.sum(y_train == pred_train)/(y_train.shape[0])
unweight_val_acc = np.sum(y_val == pred_val)/(y_val.shape[0])

weight_train_acc = 0
weight_val_acc = 0
for i in range(k_y_min,k_y+1):
        if(np.sum(y_train == i) > 0):
                        weight_train_acc = weight_train_acc + np.sum(y_train[y_train == pred_train] == i)/(np.sum(y_train == i)*(k_y+1-k_y_min))
        if(np.sum(y_val == i) > 0):
                weight_val_acc = weight_val_acc + np.sum(y_val[y_val == pred_val] == i)/(np.sum(y_val == i)*(k_y+1-k_y_min))

print(unweight_train_acc)
print(weight_train_acc)
print(unweight_val_acc)
print(weight_val_acc)
