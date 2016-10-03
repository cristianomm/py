# Code source: Gael Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import csv

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import datasets

ids = []
X = []
y = []

colNames=[
'CP(Caracteres)',
'CP(Erros)',
'CL(Vel. leitura)',
'PL(Vel. leitura)',
'PL(Total erros)',
'PL(Erros regulares)',
'PL(Erros irregulares)',
'PL(Erros pseudo)',
'PT(N Palavras)'
]
#'Compreensao leitora(Resp. espontaneas)',
#'Compreensao leitora(Resp. dirigidas)',

dataFile = '/home/cristianomm/Projects/PyCharm/py/data/Quantitativos_2014.csv'#Comp2014.csv'

#colunas que contem os dados das tarefas que foram realizadas
#columns = [0,1,2,4,5,6,11,12,13,14,15,16, 17]#13 -> c38 Vel.Leitura pala
columns = [1,2,3,4,5,6,7,8,9]
with open(dataFile, 'r') as sbjFile:
    subjects = []
    reader = csv.reader(sbjFile, delimiter=';')
    for line in reader:
        tmp = np.asarray(line)
        tmp = tmp[columns]
        X.append(np.asarray(tmp[0:], dtype=np.float32))
        ids.append(line[0])

sbjFile.close()

X = np.asarray(X)
y = np.empty(len(X))

#print(X[:,[5]])

#calcula a mediana da velocidade de leitura da task de pseudopalavras
median = np.median(X[:, [4]])

# atribui quem eh bom ou mau leitor conforme a mediana da velocidade de leitura
#for v in X[:, [4]]:
 #   if (v < median):
 #       y.append(0)
 #   else:
 #       y.append(1)

X = preprocessing.normalize(X)

#utiliza somente as colunas
selCols=[0,1,2,3,4,5,6,7,8]
X = X[:, selCols]

np.random.seed(5)
centers = [[1, 1], [-1, -1], [1, -1]]
#iris = datasets.load_iris()
#'k_means_iris_8': KMeans(n_clusters=8), 'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1, init='random')

estimators = {'k_means_2': KMeans(n_clusters=2)}


fignum = 1
for name, est in estimators.items():
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    est.fit(X)

    for i in range(len(X)):
        p = est.predict(X[i])
        y[i] = p
        #print('{0} - [{1}] >> {2}'.format(ids[i], X[i], p))

    print('Bons:{0} - Maus:{1}'.format(np.sum(y == 0), np.sum(y == 1)))

    with open('kmeans_classify.csv', 'w') as fp:
        a = csv.writer(fp)
        for i in range(len(X)):
            a.writerow([ids[i],y[i],','.join(map(str,X[i, selCols])).replace('\"', '')])


    labels = est.labels_

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel(colNames[selCols[0]])
    ax.set_ylabel(colNames[selCols[1]])
    ax.set_zlabel(colNames[selCols[2]])
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

for name, label in [('Bom', 0),
                    ('Mau', 1)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel(colNames[selCols[0]])
ax.set_ylabel(colNames[selCols[1]])
ax.set_zlabel(colNames[selCols[2]])
plt.show()