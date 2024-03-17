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

priors = [0.5]
D, L = MLlibrary.load('Train.txt')

ZD, mean, stdv = MLlibrary.ZNormalization(D)
C=np.logspace(-5, 2, num=8)
gamma = [10**(-5),10**(-4),10**(-3)]
model = SVM.SVM()
PCA8 = PCA.PCA(ZD, L, 8)
PCA7 = PCA.PCA(ZD, L, 7)

P = MLlibrary.LDA(ZD, L, 2)
LDA = np.dot(P.T, ZD)

P2 = MLlibrary.LDA(PCA7, L, 2)
PCALDA = np.dot(P2.T, PCA7)
print ("Executing RBF SVM with no re-balancing")


(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(ZD, L)
minDCF = []
print("Start RBF SVM on single fold of z normalized features")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for g in gamma:
        print("")
        print("Working with gamma:", g) 
        for value in C:
            model.train(DTRSF, LTRSF, option='RBF',C=value, gamma=g) 
            temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
            minDCF.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCFRBF(C, minDCF, "C")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA8, L)
minDCF8 = []
print("Start RBF SVM on single fold of z normalized features with PCA = 8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for g in gamma:
        print("")
        print("Working with gamma:", g) 
        for value in C:
            model.train(DTRSF, LTRSF, option='RBF',C=value, gamma=g) 
            temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
            minDCF8.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCFRBF(C, minDCF8, "C")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA7, L)
minDCF7 = []
print("Start RBF SVM on single fold of z normalized features with PCA = 7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for g in gamma:
        print("")
        print("Working with gamma:", g) 
        for value in C:
            model.train(DTRSF, LTRSF, option='RBF',C=value, gamma=g) 
            temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
            minDCF7.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCFRBF(C, minDCF7, "C")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(LDA, L)
minDCFLDA = []
print("Start RBF SVM on single fold of z normalized features with LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for g in gamma:
        print("")
        print("Working with gamma:", g) 
        for value in C:
            model.train(DTRSF, LTRSF, option='RBF',C=value, gamma=g) 
            temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
            minDCFLDA.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCFRBF(C, minDCFLDA, "C")


(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCALDA, L)
minDCFPCALDA = []
print("Start RBF SVM on single fold of z normalized features with PCA+LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for g in gamma:
        print("")
        print("Working with gamma:", g) 
        for value in C:
            model.train(DTRSF, LTRSF, option='RBF',C=value, gamma=g) 
            temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
            minDCFPCALDA.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCFRBF(C, minDCFPCALDA, "C")


minDCF5fold = []
print("Start RBF SVM with 5-fold on z normalized features")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for g in gamma:
        print("")
        print("Working with gamma:", g) 
        for value in C:
            temp = MLlibrary.KfoldSVM(ZD, L, model, option='RBF',gamma=g, C=value, prior=priors[i])
            minDCF5fold.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCFRBF(C, minDCF5fold, "C")

minDCF5fold8 = []
print("Start RBF SVM with 5-fold on z normalized features with PCA = 8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for g in gamma:
        print("")
        print("Working with gamma:", g) 
        for value in C:
            temp = MLlibrary.KfoldSVM(PCA8, L, model, option='RBF',gamma=g, C=value, prior=priors[i])
            minDCF5fold8.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCFRBF(C, minDCF5fold8, "C")

minDCF5fold7 = []
print("Start RBF SVM with 5-fold on z normalized features with PCA = 7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for g in gamma:
        print("")
        print("Working with gamma:", g) 
        for value in C:
            temp = MLlibrary.KfoldSVM(PCA7, L, model, option='RBF',gamma=g, C=value, prior=priors[i])
            minDCF5fold7.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCFRBF(C, minDCF5fold7, "C")

minDCF5foldLDA = []
print("Start RBF SVM with 5-fold on z normalized features with LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for g in gamma:
        print("")
        print("Working with gamma:", g) 
        for value in C:
            temp = MLlibrary.KfoldSVM(LDA, L, model, option='RBF',gamma=g, C=value, prior=priors[i])
            minDCF5foldLDA.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCFRBF(C, minDCF5foldLDA, "C")

minDCF5foldPCALDA = []
print("Start RBF SVM with 5-fold on z normalized features with PCA+LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for g in gamma:
        print("")
        print("Working with gamma:", g) 
        for value in C:
            temp = MLlibrary.KfoldSVM(PCALDA, L, model, option='RBF',gamma=g, C=value, prior=priors[i])
            minDCF5foldPCALDA.append(temp)
            print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCFRBF(C, minDCF5foldPCALDA, "C")