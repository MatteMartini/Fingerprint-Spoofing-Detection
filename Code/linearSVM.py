# -*- coding: utf-8 -*-
"""

@authors: Gabriele Lucca, Matteo Martini
"""

import MLlibrary
import metrics
import PCA
import classifierSVM
import matplotlib.pyplot as plt
import numpy as np

#minDCF evaluation
priors = [0.5]
D, L = MLlibrary.load('Train.txt')
ZD, mean, stdv  = MLlibrary.ZNormalization(D)
C=np.logspace(-4, -2.522878, num=30)
model = classifierSVM.SVM()
PCA8 = PCA.PCA(ZD, L, 8)
PCA7 = PCA.PCA(ZD, L, 7)

P = MLlibrary.LDA(ZD, L, 2)
LDA = np.dot(P.T, ZD)

P2 = MLlibrary.LDA(PCA7, L, 2)
PCALDA = np.dot(P2.T, PCA7)

print ("Executing linear SVM with no re-balancing")

#minDCF graph

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(ZD, L)
minDCF = []
print("Start linear SVM on single fold of z normalized features")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for value in C:
        model.train(DTRSF, LTRSF, option='linear',C=value) 
        temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        minDCF.append(temp)
        print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(C, minDCF, "C", "no PCA")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA7, L)
minDCF7 = []
print("Start linear SVM on single fold of z normalized features with PCA = 7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for value in C:
        model.train(DTRSF, LTRSF, option='linear',C=value) 
        temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        minDCF7.append(temp)
        print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(C, minDCF7, "C", "PCA = 7")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA8, L)
minDCF8 = []
print("Start linear SVM on single fold of z normalized features with PCA = 8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for value in C:
        model.train(DTRSF, LTRSF, option='linear',C=value) 
        temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        minDCF8.append(temp)
        print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(C, minDCF8, "C", "PCA = 8")


(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(LDA, L)
minDCFLDA = []
print("Start linear SVM on single fold of z normalized features with LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for value in C:
        model.train(DTRSF, LTRSF, option='linear',C=value) 
        temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        minDCFLDA.append(temp)
        print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(C, minDCFLDA, "C", "LDA")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCALDA, L)
minDCFPCALDA = []
print("Start linear SVM on single fold of z normalized features with PCA+LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for value in C:
        model.train(DTRSF, LTRSF, option='linear',C=value) 
        temp = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        minDCFPCALDA.append(temp)
        print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(C, minDCFPCALDA, "C", "PCA+LDA")
plt.figure()

MLlibrary.labels = []

print("------------K-fold K = 5-----------")

print("Linear SVM with 5-fold")

minDCF3fold = []
print("Start linear SVM with 5-fold on z normalized features")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for value in C:
        temp = MLlibrary.KfoldSVM(ZD, L, model, option='linear', C=value, prior=priors[i])
        minDCF3fold.append(temp)
        print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(C, minDCF3fold, "C", "no PCA/LDA")

print("Linear SVM with 5-fold with PCA = 7")

minDCF3fold7 = []
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for value in C:
        temp = MLlibrary.KfoldSVM(PCA7, L, model, option='linear', C=value, prior=priors[i])
        minDCF3fold7.append(temp)
        print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(C, minDCF3fold7, "C", "PCA7")

print("Linear SVM with 5-fold with PCA = 8")

minDCF3fold8 = []
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for value in C:
        temp = MLlibrary.KfoldSVM(PCA8, L, model, option='linear', C=value, prior=priors[i])
        minDCF3fold8.append(temp)
        print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(C, minDCF3fold8, "C", "PCA8")

print("Linear SVM with 5-fold with LDA")

minDCF3foldLDA = []
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for value in C:
        temp = MLlibrary.KfoldSVM(LDA, L, model, option='linear', C=value, prior=priors[i])
        minDCF3foldLDA.append(temp)
        print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(C, minDCF3foldLDA, "C", "LDA")


print("Linear SVM with 5-fold with PCA+LDA")
minDCF3foldPCALDA = []
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for value in C:
        temp = MLlibrary.KfoldSVM(PCALDA, L, model, option='linear', C=value, prior=priors[i])
        minDCF3foldPCALDA.append(temp)
        print("For C=", value, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(C, minDCF3foldPCALDA, "C", "PCA+LDA")
plt.figure()