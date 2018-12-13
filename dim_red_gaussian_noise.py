import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import imp
import matplotlib.pyplot as plt

data=np.load('../data/processed_data.npz')
#print(np.shape(data['pids'])) #PIDs
pids=data['pids']
#print(np.shape(data['X'])) #fMRI data for each task for each person
X=data['X']
#print(np.shape(data['X_sym'])) #removes duplicates
X_sym=data['X_sym']
#print(np.shape(data['Y'])) #DASS responses
Y=data['Y']
#print(np.shape(data['Z'])) #MASQ responses
Z=data['Z']

#print(np.shape(X[0,:,0]))
duplicates=10
x_train_noise=np.zeros((120*(duplicates+1),54756))
x_train=X[0:120,:,0]
x_train_noise[0:120,0:54756]=x_train
for i in range(1,duplicates+1):
	for j in range(0, 120):
		x_train_noise[j+120*i,0:54756]=x_train[i,0:54756]+np.random.normal(0, 0.1, 54756)


print(x_train_noise[:,0])

x_test=X[120:,:,0]

#x=np.reshape(x,(len(x),1))
#print(np.shape(x))

y_train=Y[0:120,:]
#y=np.reshape(y,(len(y),1))
#print(np.shape(y))
#print(y[0,:])

cca = CCA(n_components=10)
x_c, y_c = cca.fit_transform(x_train, y_train)
print('====================CCA====================')
print('Canonical Variables/Projections')
print(np.shape(x_c))
print(np.shape(y_c))
#print(x_c[0,:])
#print(y_c[0,:])
print('Weights/Bases')
print(np.shape(cca.x_weights_))
print(np.shape(cca.y_weights_))

pca = PCA(n_components=10)
x_pc = pca.fit_transform(x_train)
print('====================PCA====================')
print('Projections')
print(np.shape(x_pc))
#print(x_pc[0,:])
print('Bases')
print(np.shape(pca.components_))

n_comp=list(range(1,len(x_train_noise[:,0]),int(len(x_train_noise[:,0])/10)))
loss_array_train=[0]*len(n_comp)
loss_array_test=[0]*len(n_comp)
count=0
for i in n_comp:
	#pca_train=SparsePCA(n_components=i,normalize_components=True,random_state=0)
	pca_train = PCA(n_components=i)
	x_pc_train = pca_train.fit_transform(x_train_noise)
	x_projected_train = pca_train.inverse_transform(x_pc_train)
	loss_train=((x_train_noise - x_projected_train) ** 2).mean()
	loss_array_train[count]=loss_train
	#=================================================
	#pca_test = PCA(n_components=i)
	x_pc_test = pca_train.transform(x_test)
	x_projected_test = pca_train.inverse_transform(x_pc_test)
	loss_test=((x_test - x_projected_test) ** 2).mean()
	loss_array_test[count]=loss_test
	count=count+1
#print(loss_array)
plt.figure()
line1, =plt.plot(n_comp,loss_array_train, c='red', linewidth=2)
line2, =plt.plot(n_comp,loss_array_test, c='blue', linewidth=2)
plt.xlabel('# Dimensions')
plt.ylabel('Mean Squared Error')
plt.legend((line1, line2), ('Training Set', 'Test Set'))
plt.savefig('./output/pca_error.pdf')
#print(loss)