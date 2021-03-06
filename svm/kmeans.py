
import csv

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def plot(data, colNames, clf, refCols):
    fignum = 1
    fig = plt.figure(fignum, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    labels = clf.labels_
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel(colNames[refCols[0]])
    ax.set_ylabel(colNames[refCols[1]])
    ax.set_zlabel(colNames[refCols[2]])
    fignum = fignum + 1
    plt.show()

    return


def loadSbj(lineData):
    tmp = np.asarray(lineData)
    tmp = tmp[columns]
    X.append(np.asarray(tmp[0:], dtype=np.float32))
    ids.append(lineData[0])  # carrega o id

    # carrega a indicacao [B(om),M(au),R(egular)]
    # conforme classificacao previa a partir do exame de neuroimagem funcional
    if (lineData[1].startswith('B')):
        gd.append(0)
    elif (lineData[1].startswith('M')):
        gd.append(1)
    elif (lineData[1].startswith('R')):
        gd.append(2)
    else:
        gd.append(-1)

    return


def save_csv(ids, gd, y, X):
    with open('../data/kmeans_classify.csv', 'w') as fp:
        a = csv.writer(fp)
        for i in range(len(X)):
            a.writerow([ids[i], y[i], ','.join(map(str, X[i]))])
    return


def validate(X, y, clusters):
    y_pred = []
    y_true = []

    loo = cross_validation.LeaveOneOut(n=len(y))
    for train_index, test_index in loo:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)

        clf = KMeans(n_clusters=clusters)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)[0]
        real = y_test[0]

        y_pred.append(pred)
        y_true.append(real)

    [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fscore = 2 * TP / (2 * TP + FP + FN)

    # print('Bons:{0} - Maus:{1}'.format((y.count(0)), (y.count(1))))
    #print('Accuracy:{0} - Precision:{1} - Recall:{2} - FScore:{3}'.format(accuracy, precision, recall, fscore))
    return [accuracy, precision, recall, fscore, clf]


if __name__ == "__main__":
    ids = []
    X = []
    y = []
    gd = [] #groud truth...
    kmeansResult=[]#guarda o resultado da classificacao para o kmeans

    colNames=[
    'CP(Caracteres)',       #0
    'CP(Erros)',            #1
    'CL(Vel. leitura)',     #2
    'PL(Vel. leitura)',     #3
    'PL(Total erros)',      #4
    'PL(Erros regulares)',  #5
    'PL(Erros irregulares)',#6
    'PL(Erros pseudo)',     #7
    'PT(N Palavras)'        #8
    ]
    dataFile = '../data/Quantitativos_2014.csv'#Comp2014.csv'

    #colunas que contem os dados das tarefas que foram realizadas
    columns = [2,3,4,5,6,7,8,9,10]  #colunas com os dados
    selCols=[]  #Colunas selecionadas para realizar o agrupamento

    #indica se deve incluir na classificacao os sujeitos que
    #nao possuem classificacao previa
    notClassified = False

    np.random.seed(5)

    with open(dataFile, 'r') as sbjFile:
        subjects = []
        reader = csv.reader(sbjFile, delimiter=';')
        for line in reader:
            if(notClassified):#carrega todos
                loadSbj(line)
            elif(notClassified == False and len(line[1])>0):#carrega apenas quem ja possui classificacao
                loadSbj(line)

    sbjFile.close()


    ids = np.array(ids)
    gd = np.asarray(gd)
    y = []#np.empty(len(X))
    X = np.asarray(X)

    #print(X[:,[5]])
    #calcula a mediana da velocidade de leitura da task de pseudopalavras
    median = np.median(X[:, [3]])

    # atribui quem eh bom ou mau leitor conforme a mediana da velocidade de leitura de pseudopalavras
    for i in range(len(X)):
        v = X[i,3]
        if(gd[i] == -1 or gd[i] == 2):
            if (v < median):
                y.append(0)
            else:
                y.append(1)
        else:
            y.append(gd[i])

    #X = preprocessing.normalize(X)
    y = np.array(y).astype(np.int32)

    clusters=2

    #realiza uma classificacao com todas as colunas
    XS = X[:, [x for x in range(9)]]
    allCols = validate(XS,y,clusters)
    print 'all columns:\n{0}'.format(allCols[0:4])

    clfB=None
    clfP=None
    lessCols=[]
    bestCols=[]
    poor=[1,1,1,1]
    best=[-1,-1,-1,-1]
    for a in range(9):
        for b in range(a+1, 9):
            for c in range(b+1, 9):
                selCols=[a,b,c]
                XS = X[:, selCols]
                #print(selCols)
                val = validate(XS,y,clusters)
                if(best[0] < val[0]):
                    bestCols=selCols
                    best = val
                    clfB = val[4]
                if(poor[0]>val[0]):
                    lessCols=selCols
                    poor = val
                    clfP = val[4]

    #salva um csv com os resultados
    save_csv(ids,gd,y,X)

    print 'Best: {0} \n{1}'.format(bestCols, best[0:4])
    print 'Poor: {0} \n{1}'.format(lessCols, poor[0:4])

    plot(XS, colNames, clfB, bestCols)
    plot(XS, colNames, clfP, lessCols)

