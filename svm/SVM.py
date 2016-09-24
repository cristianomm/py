import csv
import numpy as np
import random
import math

from matplotlib import pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class SVM:
    def __init__(self, columns):
        self.columns = columns
        self.X = []
        self.y = []
        self.ids = []


    '''
        carrega o arquivo com os dados comportamentais
    '''
    def load(self, fileName):

        #data = datasets.load_iris()

        with open(fileName, 'r') as sbjFile:
            subjects = []
            reader = csv.reader(sbjFile, delimiter=';')
            for line in reader:
                tmp = np.asarray(line)
                tmp = tmp[columns]
                self.X.append(np.asarray(tmp[1:], dtype=np.float32))
                self.ids.append(line[0])

        sbjFile.close()

        self.X = np.asarray(self.X)

        median = np.median(self.X[:,[5]])

        # atribui quem eh bom ou mau leitor conforme a mediana...
        for v in self.X[:,[5]]:
            if(v < median):
                self.y.append(0)
            else:
                self.y.append(1)

        self.X = self.normalise(self.X)

        #spl = self.splitData(self.X, 30)#utiliza um percentual de 30% como conj. de treinamento


    def splitData(self, X, percnt):
        percnt = int(math.floor(len(X) * ((float(percnt) / 100))))
        y = np.asarray(random.sample(X, percnt))
        return [X, y]

    '''
        Normalia cada coluna
    '''
    def normalise(self, X):
        X = preprocessing.normalize(X)
        return X

    '''

    '''
    def classify(self):
        clf = SVC(C=0.001, kernel='rbf')
        #clf.fit(self.X, self.y)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.3)

        clf.fit(X_train, y_train)

        #clf.predict()

        plt.figure(1)
        plt.clf()
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, zorder=10, cmap=plt.cm.Paired)

        # Circle out the test data
        plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
        #clf.decision_function()

        plt.show()

        plt.axis('tight')
        x_min = self.X[:, 0].min()
        x_max = self.X[:, 0].max()
        y_min = self.X[:, 1].min()
        y_max = self.X[:, 1].max()

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-.5, 0, .5])

        plt.title('linear')



        #for i in range(10):
        #    pyplot.plot(clf.support_vectors_[:,[i]])
        #pyplot.show()
        #print clf.support_vectors_
        return clf

    '''
        Retorna uma lista com o indice das colunas que mais contribuiram para a classificacao
    '''
    def getRelevantColumns(self):

        return None

    '''
        Retorna as medidas de desempenho referentes a classificacao realizada
    '''
    def getAccMetrics(self):

        return None

if __name__ == "__main__":

    dataFile = '/home/cristianomm/Projects/PyCharm/py/data/Comp2014.csv'


    #colunas que contem os dados das tarefas que foram realizadas
    columns = [0,1,2,4,5,6,11,12,13,14,15,16]#13 -> c38 Vel.Leitura pala
    svm = SVM(columns)
    svm.load(dataFile)
    svm.classify()



