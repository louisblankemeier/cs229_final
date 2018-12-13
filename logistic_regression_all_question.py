import numpy as np
from sklearn.linear_model import LogisticRegression
import imp
import matplotlib.pyplot as plt

data=np.load('../data/processed_data.npz')
pids=data['pids']
X=data['X']
X_sym=data['X_sym']
Y=data['Y']
Z=data['Z']

m = X.shape[0]
n = X.shape[1]
ny = Y.shape[1]
nz = Z.shape[1]

x_train=X[0:120,:,:]
x_val=X[120:145,:,:]
x_test = X[145:,:]

x_train_r = x_train_r[:,1:10]
x_val_r = x_val_r[:,1:10]
x_test_r = x_test_r[:,1:10]

unweighted_train_total = []
unweighted_val_total = []
weighted_train_total = []
weighted_val_total = []

for j in range(2):
        y_train=Y[0:120,j]
        y_val = Y[120:145,j]
        y_test = Y[145:,j]
        k_y = int(np.max(y_train))

        log = LogisticRegression(C = 1e-4, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000)
        #log = LogisticRegression(C = 1e-7, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000)
        log.fit(x_train_r,y_train)

        pred_train = log.predict(x_train_r)
        pred_val = log.predict(x_val_r)

        #custom accuracy
        unweight_train_acc = np.sum(y_train == pred_train)/(y_train.shape[0])
        unweight_val_acc = np.sum(y_val == pred_val)/(y_val.shape[0])

        weight_train_acc = 0
        weight_val_acc = 0
        for i in range(0,k_y+1):
                if(np.sum(y_val==i) != 0):
                        weight_train_acc = weight_train_acc + np.sum(y_train[y_train == pred_train] == i)/(np.sum(y_train == i)*(k_y+1))
                        weight_val_acc = weight_val_acc + np.sum(y_val[y_val == pred_val] == i)/(np.sum(y_val == i)*(k_y+1))
        unweighted_train_total.append(unweight_train_acc)
        unweighted_val_total.append(unweight_val_acc)
        weighted_train_total.append(weight_train_acc)
        weighted_val_total.append(weight_val_acc)
        print(unweighted_train_total)
        print(j)

for j in range(2):
        y_train=Z[0:120,j]
        y_val = Z[120:145,j]
        y_test = Z[145:,j]
        k_y = int(np.max(y_train))

        log = LogisticRegression(C = 1e-4, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000)
        #log = LogisticRegression(C = 1e-7, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000)
        log.fit(x_train_r,y_train)

        pred_train = log.predict(x_train_r)
        pred_val = log.predict(x_val_r)

        #custom accuracy
        unweight_train_acc = np.sum(y_train == pred_train)/(y_train.shape[0])
        unweight_val_acc = np.sum(y_val == pred_val)/(y_val.shape[0])

        weight_train_acc = 0
        weight_val_acc = 0
        for i in range(1,k_y+1):
                if(np.sum(y_val==i) != 0):
                        weight_train_acc = weight_train_acc + np.sum(y_train[y_train == pred_train] == i)/(np.sum(y_train == i)*(k_y))
                        weight_val_acc = weight_val_acc + np.sum(y_val[y_val == pred_val] == i)/(np.sum(y_val == i)*(k_y))
        unweighted_train_total.append(unweight_train_acc)
        unweighted_val_total.append(unweight_val_acc)
        weighted_train_total.append(weight_train_acc)
        weighted_val_total.append(weight_val_acc)
        print(unweighted_train_total)
        print(j)

np.savetxt('unweight_train_acc.txt',unweighted_train_total)
np.savetxt('unweight_val_acc.txt',unweighted_val_total)
np.savetxt('weight_train_acc.txt',weighted_train_total)
np.savetxt('weight_val_acc.txt',weighted_val_total)


'''
print(unweight_train_acc)
print(weight_train_acc)
print(unweight_val_acc)
print(weight_val_acc)
'''
