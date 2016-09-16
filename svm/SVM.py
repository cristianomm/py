import csv
import numpy as np

from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class SVM:
    def __init__(self, columns):
        self.columns = columns
        self.subjects = []


    '''
        carrega o arquivo com os dados comportamentais
    '''
    def load(self, fileName):
        with open(fileName, 'r') as sbjFile:
            reader = csv.reader(sbjFile, delimiter=';')
            for line in reader:
                self.subjects.append(line)
        subjects = np.matrix(self.subjects)
        sbjFile.close()

        X = subjects[:, 1:]
        y = []
        ids = subjects[:,0]

    '''
        Normalia cada coluna
    '''
    def normalise(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)


    '''

    '''
    def classify(self, X, y):

        loo = cross_validation.LeaveOneOut(len(y))
        clf = SVC(C=0.1, kernel='linear')

        return loo

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


