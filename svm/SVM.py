import csv
import re
import numpy as np

from numpy import genfromtxt
from sklearn import preprocessing
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


    '''

    '''
    def classify(self):

        return None

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


