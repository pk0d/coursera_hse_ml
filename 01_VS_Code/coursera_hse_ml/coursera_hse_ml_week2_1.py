import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import scale

wXdata = pd.read_csv('wine.data', index_col=False, names = [ 'type',
                                                            'alcohol',
                                                            'malic_acid',
                                                            'ash',
                                                            'alcalinity_of_ash',
                                                            'magnesium',
                                                            'total_phenols', 
                                                            'flavanoids', 
                                                            'nonflavanoid_phenols', 
                                                            'proanthocyanins', 
                                                            'color_intensity',
                                                            'hue', 
                                                            'of_diluted_wines', 
                                                            'proline'])

wYdata = wXdata['type']

#print(wdata.describe())

#for train, test in kf.split(wdata):
#    print("%s %s" % (train, test))
#    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
#    model.fit(X_train, y_train)
#    y = pd.read_csv('wine.data', header=None, usecols=[0]).values.reshape(len(X),)
 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kMeans = list()
for k in range(1, 51):
    kn = KNeighborsClassifier(n_neighbors=k)
    kn.fit(wXdata, wYdata);
    array = cross_val_score(estimator=kn, X=wXdata, y=wYdata, cv=kf, scoring='accuracy')
    m = array.mean()
    kMeans.append(m)   
 
m = max(kMeans)
indices = [i for i, j in enumerate(kMeans) if j == m]
 
print(indices[0]+1)
print(np.round(m,decimals=2))

X_scale = scale(wXdata)

kMeans = list()
for k2 in range(1, 51):
    kn2 = KNeighborsClassifier(n_neighbors=k2)
    array2 = cross_val_score(estimator=kn2, X=X_scale, y=wYdata, cv=kf, scoring='accuracy')
    m2 = array2.mean()
    kMeans.append(m2)   
 
m2 = max(kMeans)
indices2 = [i for i, j in enumerate(kMeans) if j == m2]
 
print(indices2[0]+1)
print(np.round(m2,decimals=2))