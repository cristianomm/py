import csv


import numpy as np
from sklearn import preprocessing




ids = []
X = []
y = []
dataFile = '/home/cristianomm/Projects/PyCharm/py/data/Comp2014.csv'

#percentual dos dados a sre utilizado no treinamento
trainPcnt = .3

#conjuntos de teste e treinamento
X_train = []
X_test = []
y_train = []
y_test = []



#colunas que contem os dados das tarefas que foram realizadas
columns = [0,1,2,4,5,6,11,12,13,14,15,16, 17]#13 -> c38 Vel.Leitura (pala)
with open(dataFile, 'r') as sbjFile:
    subjects = []
    reader = csv.reader(sbjFile, delimiter=';')
    for line in reader:
        tmp = np.asarray(line)
        tmp = tmp[columns]
        X.append(np.asarray(tmp[1:], dtype=np.float32))
        ids.append(line[0])

sbjFile.close()

X = np.asarray(X)

median = np.median(X[:, [5]])

# atribui quem eh bom ou mau leitor conforme a mediana...
for v in X[:, [5]]:
    if (v < median):
        y.append(0)
    else:
        y.append(1)

#X[:,[5]] = preprocessing.normalize(X[:,[5]])
print X[:,[5]]

print preprocessing.normalize(X[:,[5]])

X = X[:, [5, 6, 7]]
