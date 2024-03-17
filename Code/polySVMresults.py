# -*- coding: utf-8 -*-
"""

@authors: Gabriele Lucca, Matteo Martini
"""

import MLlibrary
import metrics
import PCA
import SVM
import numpy as np

priors = [0.5]
D, L = MLlibrary.load('Train.txt')
ZD, mean, stdv = MLlibrary.ZNormalization(D)
C = 10**(-3)
c = 30
model = SVM.SVM()
PCA8 = PCA.PCA(ZD, L, 8)
PCA7 = PCA.PCA(ZD, L, 7)

P = MLlibrary.LDA(ZD, L, 2)
LDA = np.dot(P.T, ZD)

P2 = MLlibrary.LDA(PCA7, L, 2)
PCALDA = np.dot(P2.T, PCA7)


print ("Executing linear SVM with no re-balancing")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(ZD, L)
model.train(DTRSF, LTRSF, option='polynomial', c = c, d = 2 ,C=C)
print ("Data with with no PCA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    minDCF = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    print ("For prior ", priors[i], "the minDCF is", minDCF)
    
print("")
print("END")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA7, L)
model.train(DTRSF, LTRSF, option='polynomial', c = c, d = 2 ,C=C)
print ("Data with with PCA = 7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    minDCF = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    print ("For prior ", priors[i], "the minDCF is", minDCF)
    
print("")
print("END")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA8, L)
model.train(DTRSF, LTRSF, option='polynomial', c = c, d = 2 ,C=C)
print ("Data with with PCA = 8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    minDCF = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    print ("For prior ", priors[i], "the minDCF is", minDCF)
    
print("")
print("END")


(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(LDA, L)
model.train(DTRSF, LTRSF, option='polynomial', c = c, d = 2 ,C=C)
print ("Data with with LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    minDCF = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    print ("For prior ", priors[i], "the minDCF is", minDCF)
    
print("")
print("END")


print("Start polynomial SVM with 5-fold on z normalized features with PCA+LDA")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
 
    temp = MLlibrary.KfoldSVM(ZD, L, model, option='polynomial', c = c, d = 2 ,C=C, prior=priors[i])
    print ("For prior ", priors[i], "the minDCF is", temp)
print("")
print("END")

print("Start polynomial SVM with 5-fold on z normalized features")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
 
    temp = MLlibrary.KfoldSVM(ZD, L, model, option='polynomial', c = c, d = 2 ,C=C, prior=priors[i])
    print ("For prior ", priors[i], "the minDCF is", temp)
print("")
print("END")

print("Start polynomial SVM with 5-fold on z normalized features with PCA = 8")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
 
    temp = MLlibrary.KfoldSVM(PCA8, L, model, option='polynomial', c = c, d = 2 ,C=C, prior=priors[i])
    print ("For prior ", priors[i], "the minDCF is", temp)
print("")
print("END")

print("Start polynomial SVM with 5-fold on z normalized features with PCA = 7")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
 
    temp = MLlibrary.KfoldSVM(PCA7, L, model, option='polynomial', c = c, d = 2 ,C=C, prior=priors[i])
    print ("For prior ", priors[i], "the minDCF is", temp)
print("")
print("END")

print("Start polynomial SVM with 5-fold on z normalized features with LDA")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
 
    temp = MLlibrary.KfoldSVM(LDA, L, model, option='polynomial', c = c, d = 2 ,C=C, prior=priors[i])
    print ("For prior ", priors[i], "the minDCF is", temp)
print("")
print("END")

print("Start polynomial SVM with 5-fold on z normalized features with PCA+LDA")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
 
    temp = MLlibrary.KfoldSVM(PCALDA, L, model, option='polynomial', c = c, d = 2 ,C=C, prior=priors[i])
    print ("For prior ", priors[i], "the minDCF is", temp)
print("")
print("END")