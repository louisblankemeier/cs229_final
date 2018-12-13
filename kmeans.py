import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import imp
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

data=np.load('../data/processed_data.npz')
pids=data['pids']
X=data['X']
X_sym=data['X_sym']
Y=data['Y']
Z=data['Z']

x_train=X[0:120,:,0]
x_val=X[120:145,:,0]
x_test = X[145:,0]

#look at first question
y_train=Y[0:120,0]
y_val = Y[120:145,0]
y_test = Y[145:,0]

PLOT_COLORS = ['red', 'green', 'blue', 'orange','purple','pink']

color_array=np.zeros((len(x_train[:,0]),1))
#print(np.shape(color_array))

num_cen=6
centroids_prev=np.zeros((num_cen,len(x_train[0,:])))
centroids=np.zeros((num_cen,len(x_train[0,:])))
len_cen=len(centroids[0,:])
for i in range(0,num_cen):
	centroids[i,:]=x_train[random.randint(0,len(x_train[:,0])-1),:]
#print(centroids)

mse=1e10

epsilon=1e-3
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

pca = PCA(n_components=3)
x_pc = pca.fit_transform(x_train)
#print(x_pc)

fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(0,len(x_train[:,0])):
	ax.scatter3D(x_pc[i,0],x_pc[i,1],x_pc[i,2],c=PLOT_COLORS[int(color_array[i,0]-1)])
print(color_array[:,0])
plt.show()




