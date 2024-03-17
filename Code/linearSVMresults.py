# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 23:01:31 2023

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
ZD, mean, stdv = MLlibrary.ZNormalization(D)
C = 2*10**(-3)
model = classifierSVM.SVM()
PCA8 = PCA.PCA(ZD, L, 8)
PCA7 = PCA.PCA(ZD, L, 7)

P = MLlibrary.LDA(ZD, L, 2)
LDA = np.dot(P.T, ZD)

P2 = MLlibrary.LDA(PCA7, L, 2)
PCALDA = np.dot(P2.T, PCA7)

print ("Executing linear SVM")

# SINGLE FOLD Z NORMALIZED SVM LINEAR PCA=9
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(ZD, L)
model.train(DTRSF, LTRSF, option='linear',C=C)
print ("Data z-norm")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    minDCF = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    print ("For prior ", priors[i], "the minDCF is", minDCF)
    
print("")
print("END")

# SINGLE FOLD Z NORMALIZED SVM LINEAR PCA=8
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA8, L)
model.train(DTRSF, LTRSF, option='linear',C=C)
print ("Data with with PCA = 8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    minDCF = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    print ("For prior ", priors[i], "the minDCF is", minDCF)
    
print("")
print("END")

# SINGLE FOLD Z NORMALIZED SVM LINEAR PCA=7
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA7, L)
model.train(DTRSF, LTRSF, option='linear',C=C)
print ("Data with with PCA = 7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    minDCF = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    print ("For prior ", priors[i], "the minDCF is", minDCF)
    
print("")
print("END")

# SINGLE FOLD Z NORMALIZED SVM LINEAR LDA
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(LDA, L)
model.train(DTRSF, LTRSF, option='linear',C=C)
print ("Data with with LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    minDCF = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    print ("For prior ", priors[i], "the minDCF is", minDCF)
    
print("")
print("END")

# SINGLE FOLD Z NORMALIZED SVM LINEAR LDA
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCALDA, L)
model.train(DTRSF, LTRSF, option='linear',C=C)
print ("Data with with PCA+LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    minDCF = metrics.minimum_detection_costs(model.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    print ("For prior ", priors[i], "the minDCF is", minDCF)
    
print("")
print("END")

print("Start linear SVM with 3-fold on z normalized features ")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
 
    temp = MLlibrary.KfoldSVM(ZD, L, model, option='linear', C=C, prior=priors[i])
    print ("For prior ", priors[i], "the minDCF is", temp)
print("")
print("END")

print("Start linear SVM with 3-fold on z normalized features with PCA = 7")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
 
    temp = MLlibrary.KfoldSVM(PCA7, L, model, option='linear', C=C, prior=priors[i])
    print ("For prior ", priors[i], "the minDCF is", temp)
print("")
print("END")

print("Start linear SVM with 3-fold on z normalized features with PCA = 8")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
 
    temp = MLlibrary.KfoldSVM(PCA8, L, model, option='linear', C=C, prior=priors[i])
    print ("For prior ", priors[i], "the minDCF is", temp)
print("")
print("END")

print("Start linear SVM with 3-fold on z normalized features with PCA = LDA")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
 
    temp = MLlibrary.KfoldSVM(LDA, L, model, option='linear', C=C, prior=priors[i])
    print ("For prior ", priors[i], "the minDCF is", temp)
print("")
print("END")

print("Start linear SVM with 3-fold on z normalized features with PCA =+ LDA")

for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
 
    temp = MLlibrary.KfoldSVM(PCALDA, L, model, option='linear', C=C, prior=priors[i])
    print ("For prior ", priors[i], "the minDCF is", temp)
print("")
print("END")