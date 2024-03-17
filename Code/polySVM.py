# -*- coding: utf-8 -*-
"""

@authors: Gabriele Lucca, Matteo Martini
"""

import MLlibrary
import metrics
import PCA
import SVM
import matplotlib.pyplot as plt
import numpy as np

def plotDCFpoly(x, y, xlabel):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 - c=0', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.5 - c=1', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.5 - c=10', color='g')
    plt.plot(x, y[3*len(x): 4*len(x)], label='min DCF prior=0.5 - c=30', color='m')

    
    plt.xlim([1e-5, 1e-1])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5 - c=0", "min DCF prior=0.5 - c=1", 
                'min DCF prior=0.5 - c=10', 'min DCF prior=0.5 - c=30'])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return

priors = [0.5]
D, L = MLlibrary.load('Train.txt')
ZD, mean = MLlibrary.CNormalization(D)
c = [0, 1, 10, 30]
C=np.logspace(-5, -1, num=15)
model = SVM.SVM()
PCA8 = PCA.PCA(ZD, L, 8)
PCA7 = PCA.PCA(ZD, L, 7)

P = MLlibrary.LDA(ZD, L, 2)
LDA = np.dot(P.T, ZD)

P2 = MLlibrary.LDA(PCA7, L, 2)
PCALDA = np.dot(P2.T, PCA7)
print ("Executing polynomial SVM with no re-balancing")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(ZD, L)
minDCF = []
print("Start polynomial SVM on single fold of z normalized features")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for clittle in c:
        print("")
        print("Evaluating value of c:", clittle)
        for value in C:
            model.train(DTRSF, LTRSF, option='polynomial', c = clittle, d = 2 ,C=value) 
            temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
            minDCF.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
plotDCFpoly(C, minDCF, "C")


(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA7, L)
minDCF7 = []
print("Start polynomial SVM on single fold of z normalized features with PCA=7")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for clittle in c:
        print("")
        print("Evaluating value of c:", clittle)
        for value in C:
            model.train(DTRSF, LTRSF, option='polynomial', c = clittle, d = 2 ,C=value) 
            temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
            minDCF7.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
plotDCFpoly(C, minDCF7, "C")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA8, L)
minDCF8 = []
print("Start polynomial SVM on single fold of z noralized features with PCA= 8")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for clittle in c:
        print("")
        print("Evaluating value of c:", clittle)
        for value in C:
            model.train(DTRSF, LTRSF, option='polynomial', c = clittle, d = 2 ,C=value) 
            temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
            minDCF8.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
plotDCFpoly(C, minDCF8, "C")


(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(LDA, L)
minDCFLDA = []
print("Start polynomial SVM on single fold of z noralized features with LDA")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for clittle in c:
        print("")
        print("Evaluating value of c:", clittle)
        for value in C:
            model.train(DTRSF, LTRSF, option='polynomial', c = clittle, d = 2 ,C=value) 
            temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
            minDCFLDA.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
plotDCFpoly(C, minDCFLDA, "C")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCALDA, L)
minDCFPCALDA = []
print("Start polynomial SVM on single fold of z normalized features with PCA+LDA")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for clittle in c:
        print("")
        print("Evaluating value of c:", clittle)
        for value in C:
            model.train(DTRSF, LTRSF, option='polynomial', c = clittle, d = 2 ,C=value) 
            temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
            minDCFPCALDA.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
plotDCFpoly(C, minDCFPCALDA, "C")

# # ####----------------------k-fold

minDCF5fold = []
print("Start linear SVM with 5-fold on z normalized features")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for clittle in c:
        print("")
        print("Evaluating value of c:", clittle)
        for value in C:
            temp = MLlibrary.KfoldSVM(ZD, L, model, option='polynomial', c = clittle, d = 2 ,C=value, prior=priors[i])
            minDCF5fold.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
plotDCFpoly(C, minDCF5fold, "C")

minDCF5fold8 = []
print("Start linear SVM with 5-fold on z normalized features pca8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for clittle in c:
        print("")
        print("Evaluating value of c:", clittle)
        for value in C:
            temp = MLlibrary.KfoldSVM(PCA8, L, model, option='polynomial', c = clittle, d = 2 ,C=value, prior=priors[i])
            minDCF5fold8.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
plotDCFpoly(C, minDCF5fold8, "C")


minDCF5fold7 = []
print("Start linear SVM with 5-fold on z normalized features pca7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for clittle in c:
        print("")
        print("Evaluating value of c:", clittle)
        for value in C:
            temp = MLlibrary.KfoldSVM(PCA7, L, model, option='polynomial', c = clittle, d = 2 ,C=value, prior=priors[i])
            minDCF5fold7.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
plotDCFpoly(C, minDCF5fold7, "C")


minDCF5foldLDA = []
print("Start linear SVM with 5-fold on z normalized features w/lda")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for clittle in c:
        print("")
        print("Evaluating value of c:", clittle)
        for value in C:
            temp = MLlibrary.KfoldSVM(LDA, L, model, option='polynomial', c = clittle, d = 2 ,C=value, prior=priors[i])
            minDCF5foldLDA.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
plotDCFpoly(C, minDCF5foldLDA, "C")

minDCF5foldPCALDA = []
print("Start linear SVM with 5-fold on z normalized features w/ PCA + LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for clittle in c:
        print("")
        print("Evaluating value of c:", clittle)
        for value in C:
            temp = MLlibrary.KfoldSVM(PCALDA, L, model, option='polynomial', c = clittle, d = 2 ,C=value, prior=priors[i])
            minDCF5foldPCALDA.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
plotDCFpoly(C, minDCF5foldPCALDA, "C")