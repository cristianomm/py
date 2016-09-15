import csv
import re
import numpy as np

from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

if __name__ == "__main__":
    fileName = 'C:\\Disco\\Users\\Cristiano\\Google Drive\\Trabalho\\PRAIAS\\Dados\\Comportamentais\\Comp2014.csv'

    columns = [0,1,2,4,5,6,11,12,13,14,15,16]
    formats=[str, float, float, float, float, float, float, float, float, float, float, float]
#

#def load(fileName):
    subjects = []

    with open(fileName, 'r') as sbjFile:
        reader = csv.reader(sbjFile, delimiter=';')
        for line in reader:
            subjects.append(line)
    subjects = np.matrix(subjects)
    sbjFile.close()
    print subjects[:,[columns]]




    #return subjects
