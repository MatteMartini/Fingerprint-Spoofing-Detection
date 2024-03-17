import MLlibrary
import PCA
import metrics
import LogisticRegression
import GMM
import GaussianClassifier
import GMMDiag
import GMMTiedCov
import numpy as np
import matplotlib.pyplot as plt
import scipy

def vcol(v):
    return v.reshape((v.size, 1))
def vrow(v):
    return v.reshape((1, v.size))


def SbSw(D, L):
    SB = 0
    SW = 0
    mu = vcol(D.mean(1))  #media di tutti i sample di una riga per ogni attributo, è un vettore di 4 elementi in questo caso, perche ci sono 4 possibili attributi => è la media del dataset, cioe la media di tutti i sample, distinto per ogni attributo!
    for i in range(L.max() + 1): #L.max() +1 ti da il numero di classi differenti tra i dati passati
        DCls = D[:, L == i]  #ti prendi cosi tutti i sample della classe in analisi, e saranno ad classe 0, poi classe 1 ecc.
        muCls = vcol(DCls.mean(1)) #media degli elementi di una classe! Grazie alla riga prima escludi gli elementi della classe in analisi
        SW += np.dot(DCls - muCls, (DCls - muCls).T)  #ad ogni iterazione aggiungi il contributo della parte a destra
        SB += DCls.shape[1] * np.dot(muCls - mu, (muCls - mu).T)  #DCls.shape[1] corrisponde al nc della formula

    SW /= D.shape[1] # è il fratto N che sta nelle 2 formule
    SB /= D.shape[1]
    return SB, SW

def LDA(D, L, m):
    SB, SW = SbSw(D, L)
    s, U = scipy.linalg.eigh(SB, SW) 
    return U[:, ::-1][:, 0:m]


priors = [0.5]
D, L = MLlibrary.load('Train.txt')
ZD, mean, standardDeviation = MLlibrary.ZNormalization(D)
PCA7=PCA.PCA(ZD, L, 7)
PCA8 = PCA.PCA(ZD, L, 8)
PCA9 = PCA.PCA(ZD, L, 9)
P2=LDA(D, L, 2)
transformed_data2=np.dot(P2.T, D)

P2L=LDA(PCA7, L, 2)
transformed_data=np.dot(P2L.T, PCA7)


gc = GaussianClassifier.GaussianClassifier()
lr = LogisticRegression.LogisticRegression()
lambd = 1e-2
numberOfSplitToPerform = 2
gmm = GMM.GMM()
gmm3= GMMTiedCov.GMMTiedCov()
gmm2= GMMDiag.GMMDiag()
numberOfPoints=13
effPriorLogOdds = np.linspace(-6, 6, numberOfPoints)
effPriors = 1/(1+np.exp(-1*effPriorLogOdds))

print("Start Full-Cov RAW with 5-fold  PCA m=7") #0.329   =>   0.678    VA CALIBRATO
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldActualDCF(PCA7, L, gc, prior=priors[i])
    print("Actual DCF MVG Tied-cov with prior=%.1f:  %.3f" %(priors[i], actualDCF))
print("")
print("END")


print("Start Linear Logistic Regression with 5-fold RAW features PCA + LDA with lambda=10^(-2) and pi_T=0.5")  #0.421   =>   1.014
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldLRActualDCF(transformed_data, L, lr, lambd, prior=priors[i])
    print("Actual DCF Linear Logistic Regression with prior=%.1f:  %.3f" %(priors[i], actualDCF))
print("")
print("END")


print("Start Tied-Cov 8 GMM components with 5-fold on z normalized features")   #0.225  =>   0.487   DA CALIBRARE
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldGMMActualDCF(ZD, L, gmm3, numberOfSplitToPerform, prior=priors[i])
    print("Actual DCF Full-Cov 8 GMM components with prior=%.1f:  %.3f" %(priors[i], actualDCF))
print("")
print("END")

print("Start Naive Bayes 4 GMMDiag components with 5-fold Raw")   #0.225  =>   0.487   DA CALIBRARE
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldGMMActualDCF(PCA8, L, gmm2, numberOfSplitToPerform, prior=priors[i])
    print("Actual DCF Naive Bayes 4 GMMDiag components with prior=%.1f:  %.3f" %(priors[i], actualDCF))
print("")
print("END")


#FULL-COV RAW
actualDCFs = []
minDCFs = []
for i in range(numberOfPoints):
    actualDCFs.append(MLlibrary.KfoldActualDCF(PCA7, L, gc, prior=effPriors[i]))
    minDCFs.append(MLlibrary.Kfold(PCA7, L, gc, prior=effPriors[i]))
    print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
MLlibrary.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "Full-Cov")

# #LINEAR LOGISTIC REGRESSION RAW
actualDCFs = []
minDCFs = []
for i in range(numberOfPoints):
    actualDCFs.append(MLlibrary.KfoldLRActualDCF(transformed_data, L, lr, lambd, prior=effPriors[i]))
    minDCFs.append(MLlibrary.KfoldLR(transformed_data, L, lr, lambd, prior=effPriors[i]))
    print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
MLlibrary.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "Logistic Regression")


#GMM Z-Norm 8 components 
actualDCFs = []
minDCFs = []
for i in range(numberOfPoints):
    actualDCFs.append(MLlibrary.KfoldGMMActualDCF(ZD, L, gmm3, numberOfSplitToPerform, prior=effPriors[i]))
    minDCFs.append(MLlibrary.KfoldGMM(ZD, L, gmm3, numberOfSplitToPerform, prior=effPriors[i]))
    print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
MLlibrary.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "GMM")


#Score calibration on Full Cov
actualDCFs0 = []
actualDCFs1 = []
minDCFs = []
for i in range(numberOfPoints):
    print("Working on point:", i)
    minDCFs.append(MLlibrary.Kfold(PCA7, L, gc, prior=effPriors[i]))
    actualDCFs0.append(MLlibrary.KfoldActualDCFCalibrated(PCA7, L, gc, lambd=1e-3, prior=effPriors[i]))
    actualDCFs1.append(MLlibrary.KfoldActualDCFCalibrated(PCA7, L, gc, lambd=1e-2, prior=effPriors[i]))
MLlibrary.bayesErrorPlotV2(actualDCFs0, actualDCFs1, minDCFs, effPriorLogOdds, "Full Cov", "10^(-3)", "10^(-2)")



#Score calibration on Logistic Regression
actualDCFs0 = []
actualDCFs1 = []
minDCFs = []
for i in range(numberOfPoints):
    print("Working on point:", i)
    minDCFs.append(MLlibrary.KfoldLR(transformed_data, L, lr, lambd, prior=effPriors[i]))
    actualDCFs0.append(MLlibrary.KfoldLRActualDCFCalibrated(transformed_data, L, lr, lambd, lambd2=1e-3, prior=effPriors[i]))
    actualDCFs1.append(MLlibrary.KfoldLRActualDCFCalibrated(transformed_data, L, lr, lambd, lambd2=1e-2, prior=effPriors[i]))
MLlibrary.bayesErrorPlotV2(actualDCFs0, actualDCFs1, minDCFs, effPriorLogOdds, "Logistic Regression", "10^(-3)", "10^(-2)")



#Score calibration on GMM Z-Norm 8 components
actualDCFs0 = []
actualDCFs1 = []
minDCFs = []
for i in range(numberOfPoints):
    print("Working on point:", i)
    minDCFs.append(MLlibrary.KfoldGMM(ZD, L, gmm3, numberOfSplitToPerform, prior=effPriors[i]))
    actualDCFs0.append(MLlibrary.KfoldGMMActualDCFCalibrated(ZD, L, gmm3, numberOfSplitToPerform, lambd=1e-3, prior=effPriors[i]))
    actualDCFs1.append(MLlibrary.KfoldGMMActualDCFCalibrated(ZD, L, gmm3, numberOfSplitToPerform, lambd=1e-2, prior=effPriors[i]))
MLlibrary.bayesErrorPlotV2(actualDCFs0, actualDCFs1, minDCFs, effPriorLogOdds, "GMM", "10^(-3)", "10^(-2)")



print("Start Full-Cov RAW with 5-fold  PCA m=7 SCORES CALIBRATED")
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldActualDCFCalibrated(PCA7, L, gc, lambd, prior=priors[i])
    print("Actual DCF Full-Cov with prior=%.1f:  %.3f after score calibration" %(priors[i], actualDCF))
print("")
print("END")



print("Start Linear Logistic Regression with 5-fold RAW features PCA + LDA with lambda=10^(-2) and pi_T=0.5 SCORES CALIBRATED")
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldLRActualDCFCalibrated(transformed_data, L, lr, lambd, lambd2=1e-2, prior=priors[i])
    print("Actual DCF Linear Logistic Regression with prior=%.1f:  %.3f after score calibration" %(priors[i], actualDCF))
print("")
print("END")



print("Start  Naive Bayes 4 GMMDiag components with 5-fold Raw SCORES CALIBRATED")
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldGMMActualDCFCalibrated(PCA8, L, gmm2, numberOfSplitToPerform, lambd, prior=priors[i])
    print("Actual Naive Bayes 4 GMMDiag components with prior=%.1f:  %.3f after score calibration" %(priors[i], actualDCF))
print("")
print("END")