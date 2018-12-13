import numpy as np
from sklearn.decomposition import PCA
import imp
import matplotlib.pyplot as plt

data=np.load('../data/processed_data.npz')
pids=data['pids']
X=data['X']
X_sym=data['X_sym']
Y=data['Y']
Z=data['Z']

x_train=X[0:120,:,0]
x_test=X[120:,:,0]

y_train=Y[0:120,:]

n_comp=list(range(1,120,5))
loss_array_train=[0]*len(n_comp)
loss_array_test=[0]*len(n_comp)
energy = [0]*len(n_comp)
count=0
for i in n_comp:
	pca_train = PCA(n_components=i)
	x_pc_train = pca_train.fit_transform(x_train)
	energy[count] = np.sum(pca_train.explained_variance_ratio_)
	x_projected_train = pca_train.inverse_transform(x_pc_train)
	loss_train=((x_train - x_projected_train) ** 2).mean()
	loss_array_train[count]=loss_train
	#=================================================
	x_pc_test = pca_train.transform(x_test)
	x_projected_test = pca_train.inverse_transform(x_pc_test)
	loss_test=((x_test - x_projected_test) ** 2).mean()
	loss_array_test[count]=loss_test
	count=count+1
plt.figure()
line1, =plt.plot(n_comp,loss_array_train, c='red', linewidth=2)
line2, =plt.plot(n_comp,loss_array_test, c='blue', linewidth=2)
plt.xlabel('# Dimensions')
plt.ylabel('Mean Squared Error')
plt.legend((line1, line2), ('Training Set', 'Test Set'))
plt.savefig('./output/pca_error.pdf')
plt.figure()
plt.plot(n_comp,energy)
plt.savefig('./output/pca_energy.pdf')
print(n_comp)
print(energy)
