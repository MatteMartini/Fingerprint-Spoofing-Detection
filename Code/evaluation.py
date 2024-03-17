# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 01:29:05 2023

@authors: Gabriele Lucca, Matteo Martini
"""

import MLlibrary
import GaussianClassifier
import GaussianClassifierNB
import GaussianClassifierTiedCov
import LogisticRegression
import SVM
import metrics
import PCA
import matplotlib.pyplot as plt
import numpy as np

priors = [0.5]
D, L = MLlibrary.load('Train.txt')
DT, LT = MLlibrary.load('Test.txt')
ZD, mean, standardDeviation = MLlibrary.ZNormalization(D)
ZDT, mean, standardDeviation = MLlibrary.ZNormalization(DT)

D8 = PCA.PCA(ZD, L, 8)
DT8 = PCA.PCA(ZDT, L, 8)

print ("----MVG Full Cov----")
model = GaussianClassifier.GaussianClassifier()
model.train(ZD, L)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10)  
    print("min DCF MVG Full-Cov with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
    
print ("----MVG Full Cov with PCA=8----")
model = GaussianClassifier.GaussianClassifier()
model.train(D8, L)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10)  
    print("min DCF MVG Full-Cov with PCA=8 with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
    
print ("----MVG Diag Cov----")
model = GaussianClassifierNB.GaussianClassifierNB()
model.train(ZD, L)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10)  
    print("min DCF MVG Diag Cov with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
    
print ("----MVG Diag Cov with PCA=8----")
model = GaussianClassifierNB.GaussianClassifierNB()
model.train(D8, L)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10)  
    print("min DCF MVG Diag-Cov with PCA=8 with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
    
print ("----MVG Tied Cov----")
model = GaussianClassifierTiedCov.GaussianClassifierTiedCov()
model.train(ZD, L)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10)  
    print("min DCF MVG Tied-Cov with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
    
print ("----MVG Tied Cov with PCA=8----")
model = GaussianClassifierTiedCov.GaussianClassifierTiedCov()
model.train(D8, L)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10)  
    print("min DCF MVG Tied-Cov with PCA=8 with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
    
lambd=10**(-2)
print ("----Logistic Regression----")
model = LogisticRegression.LogisticRegression()
model.train(ZD, L, lambd, 0.5)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10)  
    print("min DCF Logistic Regression with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
    
lambd=10**(-2)
print ("----Logistic Regression with PCA m=8----")
model = LogisticRegression.LogisticRegression()
model.train(D8, L, lambd, 0.5)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10)  
    print("min DCF Logistic Regression with PCA m=8 with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

C=2*10**(-5)
print ("----Linear SVM----")
model = SVM.SVM()
model.train (ZD, L, option='linear', C=C)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10)  
    print("min DCF Linear SVM with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
    
C=2*10**(-5)
print ("----Linear SVM with PCA m=8----")
model = SVM.SVM()
model.train (D8, L, option='linear', C=C)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10)  
    print("min DCF linear SVM with PCA m=8 with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

C=2*10**(-5)
print ("----Quadratic SVM----")
model = SVM.SVM()
model.train (ZD, L, option='polynomial', d=2, c=30, C=C)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10)  
    print("min DCF Quadratic SVM with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

C=2*10**(-5)
print ("----Quadratic SVM with PCA m=8----")
model = SVM.SVM()
model.train (D8, L, option='polynomial', d=2, c=30, C=C)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10)  
    print("min DCF Quadratic SVM with PCA m=8 with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

C=10**(-4)
print ("----RBF SVM----")
model = SVM.SVM()
model.train (ZD, L, option='RBF', C=C, gamma=10**(-5))
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10)  
    print("min DCF RBF SVM with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

C=10**(-4)
print ("----RBF SVM with PCA m=8----")
model = SVM.SVM()
model.train (D8, L, option='RBF', C=C, gamma=10**(-5))
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10)  
    print("min DCF RBF SVM with PCA m=8 with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
    

import MLlibrary
import GMM
import GMMTiedCov
import GMMDiag
import metrics
import PCA
import matplotlib.pyplot as plt
import numpy as np
priors = [0.5]
D, L = MLlibrary.load('Train.txt')
DT, LT = MLlibrary.load('Test.txt')
ZD, mean, standardDeviation = MLlibrary.ZNormalization(D)
ZDT, mean, standardDeviation = MLlibrary.ZNormalization(DT)
D8 = PCA.PCA(ZD, L, 8)
DT8 = PCA.PCA(ZDT, L, 8)

print ("Models without PCA")
print ("GMM with Full Cov 2 components")
model = GMM.GMM()
model.train(ZD, L, 1)
for i in range(len(priors)):
    
    #scores = MLlibrary.calibrateScores(model.predictAndGetScores(DT8), LT, lambd).flatten()
    
    #minDCFSF=metrics.minimum_detection_costs(scores, LT, priors[i], 1, 1) 
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Full-Cov model with 2 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Full Cov 4 components")
model = GMM.GMM()
model.train(ZD, L, 2)
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Full-Cov model with 4 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Full Cov 8 components")
model = GMM.GMM()
model.train(ZD, L, 3)
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Full-Cov model with 8 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Tied Cov 2 components")
model = GMMTiedCov.GMMTiedCov()
model.train(ZD, L, 1)
# GMM with Tied Cov 2 components
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Tied-Cov model with 2 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Tied Cov 4 components")
model = GMMTiedCov.GMMTiedCov()
model.train(ZD, L, 2)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Tied-Cov model with 4 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Tied Cov 8 components")
model = GMMTiedCov.GMMTiedCov()
model.train(ZD, L, 3)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Tied-Cov model with 8 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Diag Cov 2 components")
model = GMMDiag.GMMDiag()
model.train(ZD, L, 1)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Diag-Cov model with 2 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Diag Cov 4 components")
model = GMMDiag.GMMDiag()
model.train(ZD, L, 2)
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Diag-Cov model with 4 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Diag Cov 8 components")
model = GMMDiag.GMMDiag()
model.train(ZD, L, 3)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Diag-Cov model with 8 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
    
print ("Models with PCA")
print ("GMM with Full Cov 2 components")
model = GMM.GMM()
model.train(D8, L, 1)
for i in range(len(priors)):
    
    #scores = MLlibrary.calibrateScores(model.predictAndGetScores(DT8), LT, lambd).flatten()
    
    #minDCFSF=metrics.minimum_detection_costs(scores, LT, priors[i], 1, 1) 
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Full-Cov model with 2 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Full Cov 4 components")
model = GMM.GMM()
model.train(D8, L, 2)
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Full-Cov model with 4 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Full Cov 8 components")
model = GMM.GMM()
model.train(D8, L, 3)
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Full-Cov model with 8 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Tied Cov 2 components")
model = GMMTiedCov.GMMTiedCov()
model.train(D8, L, 1)
# GMM with Tied Cov 2 components
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Tied-Cov model with 2 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Tied Cov 4 components")
model = GMMTiedCov.GMMTiedCov()
model.train(D8, L, 2)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Tied-Cov model with 4 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Tied Cov 8 components")
model = GMMTiedCov.GMMTiedCov()
model.train(D8, L, 3)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Tied-Cov model with 8 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Diag Cov 2 components")
model = GMMDiag.GMMDiag()
model.train(D8, L, 1)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Diag-Cov model with 2 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Diag Cov 4 components")
model = GMMDiag.GMMDiag()
model.train(D8, L, 2)
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Diag-Cov model with 4 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Diag Cov 8 components")
model = GMMDiag.GMMDiag()
model.train(D8, L, 3)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(DT8), LT, priors[i], 1, 10) 
    
    print("min DCF of GMM Diag-Cov model with 8 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
