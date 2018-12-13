import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import imp
import matplotlib.pyplot as plt

data=np.load('../data/processed_data.npz')
pids=data['pids']
X=data['X']
X_sym=data['X_sym']
quest = np.load('../data/clustered_assignments.npz')
Y = quest['classes']

m = X.shape[0]
n = X.shape[1]

x_train=X[0:120,:,:]
x_val=X[120:145,:,:]
x_test = X[145:,:]

x_train_r = np.reshape(x_train,[120,n*14],order='F')
x_val_r = np.reshape(x_val,[145-120,n*14],order='F')
x_test_r = np.reshape(x_test,[m-145,n*14],order='F')



#look at first question
y_train=Y[0:120,0]
y_val = Y[120:145,0]
y_test = Y[145:,0]
k_y_min = int(np.min(Y))
k_y = int(np.max(Y))

log = LogisticRegression(C = 1e-1, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000)
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
        if(np.sum(y_train[y_train == pred_train] == i) > 0):
                        weight_train_acc = weight_train_acc + np.sum(y_train[y_train == pred_train] == i)/(np.sum(y_train == i)*(k_y+1-k_y_min))
        if(np.sum(y_val[y_val == pred_val] == i) > 0):
                weight_val_acc = weight_val_acc + np.sum(y_val[y_val == pred_val] == i)/(np.sum(y_val == i)*(k_y+1-k_y_min))

print(unweight_train_acc)
print(weight_train_acc)
print(unweight_val_acc)
print(weight_val_acc)
