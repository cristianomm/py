import csv
import numpy as np

from sklearn import cross_validation
from sklearn.svm import SVC
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
            reader = csv.reader(sbjFile, delimiter=';')
            for line in reader:
                tmp = np.asarray(line)
                tmp = tmp[columns]
                self.X.append(np.asarray(tmp[1:], dtype=np.float32))
                self.ids.append(line[0])

        sbjFile.close()

        self.y = [
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,0
        ]

    '''
        Normalia cada coluna
    '''
    def normalise(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)


    '''

    '''
    def classify(self):
        #normalise()
        clf = SVC(C=0.1, kernel='linear')
        clf.fit(self.X, self.y)
        print clf.support_vectors_
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
    fileName = '/home/cristianomm/Projects/PyCharm/py/data/Comp2014.csv'

    #colunas que contem os dados das tarefas que foram realizadas
    columns = [0,1,2,4,5,6,11,12,13,14,15,16]
    svm = SVM(columns)
    svm.load(fileName)
    svm.classify()



