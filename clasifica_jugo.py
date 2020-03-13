#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split

import sklearn.metrics
import sklearn.tree


data = pd.read_csv('OJ.csv')

# Remueve datos que no se van a utilizar
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)

# Crea un nuevo array que sera el target, 0 si MM, 1 si CH
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0

data['Target'] = purchasebin

# Borra la columna Purchase
data = data.drop(['Purchase'],axis=1)

# Crea un dataframe con los predictores
predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')
print(predictors)

predictores = np.array(data[predictors])
target = purchasebin

X_train, X_test, y_train, y_test = train_test_split(predictores, target, test_size=0.50)

#clf = sklearn.tree.DecisionTreeClassifier(max_depth=1)


#clf.fit(predictores, target)



#plt.figure(figsize=(10,10))
#_= sklearn.tree.plot_tree(clf)

#clf.predict(predictores)


def compute_f_1(X, Y,depth):
    n_points = len(Y)
    # esta es la clave del bootstrapping: la seleccion de indices de "estudiantes"
    indices = np.random.choice(np.arange(n_points), n_points)
    new_X = X[indices, :]
    new_Y = Y[indices]
    clf = sklearn.tree.DecisionTreeClassifier(max_depth=depth)
    clf.fit(new_X, new_Y)
    y_predict = clf.predict(new_X)
    featu = clf.feature_importances_
        
    f1 = sklearn.metrics.f1_score(new_Y, y_predict)
#    regresion = sklearn.linear_model.LinearRegression()
#   regresion.fit(new_X, new_Y)
    return {'f1':f1, 'featu':featu}

#    return f1

depth = np.arange(1,11)
f_1_mean = []
f_1_sd = []

f_1_mean_test = []
f_1_sd_test = []

mean_feature = []




for j in depth:

    n_intentos = 100
    f_1 = np.ones([n_intentos])
#    feature = np.ones([n_intentos])
    for i in range(n_intentos):
        results = compute_f_1(X_train, y_train, j)        
        f_1[i] = results['f1']
        
    f_1_mean.append(f_1.mean())
    f_1_sd.append(f_1.std())
    

    
    
for j in depth:

    n_intentos = 100
    f_1 = np.ones([n_intentos])
    feature = np.ones([n_intentos,14])
    for i in range(n_intentos):
        results = compute_f_1(X_test, y_test, j)        
        f_1[i] = results['f1']
        feature[i,:] = results['featu']        
#        f_1[i] = compute_f_1(X_test, y_test, j)
    f_1_mean_test.append(f_1.mean())
    f_1_sd_test.append(f_1.std())
    
    mean_feature.append(feature.mean(axis=0))
    
mean_feature = np.array(mean_feature)
    
    
    
    
    
    
    
    
plt.figure()
plt.errorbar(depth,f_1_mean,f_1_sd, fmt = '-o', label = "train")
plt.errorbar(depth,f_1_mean_test,f_1_sd_test, fmt = '-o', label = "test")
plt.ylabel("f_1_average_score")
plt.xlabel("Maximum depth")
plt.legend()
plt.savefig("F1_training_test.png")


plt.figure()
for i in range(14):
    plt.plot(depth, mean_feature[:,i])
plt.ylabel("feature_average")
plt.xlabel("Maximum depth")
plt.savefig("features.png")



