# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:19:57 2023

@authors: Gabriele Lucca, Matteo Martini
"""

import LogisticRegression
import PCA
import GaussianClassifierTiedCov
import GaussianClassifier
import GMMDiag
import GMMTiedCov
import numpy as np
import MLlibrary

prior=0.5
D, L = MLlibrary.load('Train.txt')
D = MLlibrary.centerData(D)
PCA7 = PCA.PCA(D, L, 7)
PCA8 = PCA.PCA(D, L, 8)

DT, LT = MLlibrary.load('Test.txt')
PCA7T = PCA.PCA(DT, LT, 7)
PCA8T = PCA.PCA(DT, LT, 8)

ZD, mean, standardDeviation = MLlibrary.ZNormalization(D)
ZDT, mean, standardDeviation = MLlibrary.ZNormalization(DT)


mvg = GaussianClassifier.GaussianClassifier()
tc = GaussianClassifierTiedCov.GaussianClassifierTiedCov()
lr = LogisticRegression.LogisticRegression()
gmmTC = GMMTiedCov.GMMTiedCov()
gmmNB = GMMDiag.GMMDiag()

lambd = 1e-2
numberOfSplitToPerformTC = 3
numberOfSplitToPerformNB = 2


print("Start Full-Cov on RAW features PCA m=7")
FPR = []
TPR = []
mvg.train(PCA7, L)
un_scores = mvg.predictAndGetScores(PCA7T)
scores = MLlibrary.calibrateScores(un_scores, LT, 1e-2).flatten()
sortedScores=np.sort(scores)
for t in sortedScores:
    m = MLlibrary.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(
        prior, 1, 10, scores, LT, t)
    FPRtemp, TPRtemp = MLlibrary.computeFPRTPR(prior, 1, 10, m)
    FPR.append(FPRtemp)
    TPR.append(TPRtemp)
print("End Full-Cov")


print("Start Linear Logistic Regression RAW with lambda=10^(-2) no PCA/LDA")
FPR1 = []
TPR1 = []
lr.train(D, L, lambd, prior)
scores = lr.predictAndGetScores(DT)
scores = MLlibrary.calibrateScores(scores, LT, 1e-2).flatten()
sortedScores=np.sort(scores)
for t in sortedScores:
    m = MLlibrary.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(
        prior, 1, 10, scores, LT, t)
    FPRtemp, TPRtemp = MLlibrary.computeFPRTPR(prior, 1, 10, m)
    FPR1.append(FPRtemp)
    TPR1.append(TPRtemp)
print("End logistic regression")


print("Start Tied-Cov 8 GMM components on z normalized features no PCA/LDA")
FPR2 = []
TPR2 = []
gmmTC.train(ZD, L, numberOfSplitToPerformTC)
scores = gmmTC.predictAndGetScores(ZDT)
scores = MLlibrary.calibrateScores(scores, LT, 1e-2).flatten()
sortedScores=np.sort(scores)
for t in sortedScores:
    m = MLlibrary.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(
        prior, 1, 10, scores, LT, t)
    FPRtemp, TPRtemp = MLlibrary.computeFPRTPR(prior, 1, 10, m)
    FPR2.append(FPRtemp)
    TPR2.append(TPRtemp)
print("End Tied-Cov GMM")

print("Start Naive Bayes 4 GMM components on RAW features PCA m=8")
FPR3 = []
TPR3 = []
gmmNB.train(PCA8, L, numberOfSplitToPerformNB)
scores = gmmNB.predictAndGetScores(PCA8T)
scores = MLlibrary.calibrateScores(scores, LT, 1e-2).flatten()
sortedScores=np.sort(scores)
for t in sortedScores:
    m = MLlibrary.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(
        prior, 1, 10, scores, LT, t)
    FPRtemp, TPRtemp = MLlibrary.computeFPRTPR(prior, 1, 10, m)
    FPR3.append(FPRtemp)
    TPR3.append(TPRtemp)
print("End Naive Bayes GMM")

MLlibrary.plotROC(FPR, TPR, FPR1, TPR1, FPR2, TPR2, FPR3, TPR3)
