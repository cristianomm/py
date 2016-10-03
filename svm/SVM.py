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

        with open(fileName, 'r') as sbjFile:
            subjects = []
            reader = csv.reader(sbjFile)
            for line in reader:
                tmp = np.asarray(line)
                tmp = tmp[columns]
                self.X.append(np.asarray(tmp[2:], dtype=np.float32))
                self.y.append(line[1])
                self.ids.append(line[0])

        sbjFile.close()

        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)



    #def splitData(self, X, percnt):
     #   percnt = int(math.floor(len(X) * ((float(percnt) / 100))))
     #   return [X, y]


    '''
        Normalia cada coluna
    '''
    def normalise(self, X):
        X = preprocessing.normalize(X)
        return X

    '''

    '''
    def classify(self):
        clf = SVC(C=0.001, kernel='linear')
        #clf.fit(self.X, self.y)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2)

        clf.fit(X_train, y_train)

        #clf.predict()

        plt.figure(1)
        plt.clf()
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, zorder=10, cmap=plt.cm.Paired)

        # Circle out the test data
        #plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
        #clf.decision_function()

        #circle support vectors
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                    s=80, facecolors='none', color='green')

        # get the separating hyperplane
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-1, 1)
        yy = a * xx - (clf.intercept_[0]) / w[1]

        # plot the parallels to the separating hyperplane that pass through the
        # support vectors
        b = clf.support_vectors_[0]
        yy_down = a * xx + (b[1] - a * b[0])
        b = clf.support_vectors_[-1]
        yy_up = a * xx + (b[1] - a * b[0])

        # plot the line, the points, and the nearest vectors to the plane
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')


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

    dataFile = '/home/cristianomm/Projects/PyCharm/py/data/kmeans_classify.csv'


    #colunas que contem os dados das tarefas que foram realizadas
    columns = [0,1,2,3,4,5,6,7,8,9]#13 -> c38 Vel.Leitura pala
    svm = SVM(columns)
    svm.load(dataFile)
    svm.classify()



