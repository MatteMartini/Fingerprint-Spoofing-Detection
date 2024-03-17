import numpy as np
import GMM
import GMMDiag
import GMMTiedCov
import metrics
import PCA
import MLlibrary
import scipy
import matplotlib.pyplot as plt

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
PCA7 = PCA.PCA(ZD, L, 7)
PCA8 = PCA.PCA(ZD, L, 8)
P2=LDA(D, L, 2)
transformed_data2=np.dot(P2.T, D)

P2L=LDA(PCA7, L, 2)
transformed_data=np.dot(P2L.T, PCA7)

numberOfSplitToPerform = 5
numberOfComponent = 3

plt.figure()
#Full Cov
gmm = GMM.GMM()
minDCF5fold = []
print("Start GMM with 5-fold on z normalized features ")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMM(ZD, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5fold.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5fold, "GMM components", "Raw")


gmm = GMM.GMM()
minDCF5foldPCA7 = []
print("Start GMM with 5-fold on z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMM(PCA7, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldPCA7.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldPCA7, "GMM components", "PCA=7")


gmm = GMM.GMM()
minDCF5foldPCA8 = []
print("Start GMM with 5-fold on z normalized features PCA m=8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMM(PCA8, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldPCA8.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldPCA8, "GMM components", "PCA=8")


gmm = GMM.GMM()
minDCF5foldLDA = []
print("Start GMM with 5-fold on z normalized features LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMM(transformed_data2, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldLDA.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldLDA, "GMM components", "LDA")


gmm = GMM.GMM()
minDCF5foldPCALDA = []
print("Start GMM with 5-fold on z normalized features PCA+LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMM(transformed_data, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldPCALDA.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldPCALDA, "GMM components", "PCA+LDA")





plt.figure()
MLlibrary.labels=[]
#Naive Bayes

gmm = GMMDiag.GMMDiag()
minDCF5fold = []
print("Start GMM with 5-fold on z normalized features ")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMM(ZD, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5fold.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5fold, "GMM components", "Raw")



gmm = GMMDiag.GMMDiag()
minDCF5foldDiagPCA7 = []
print("Start GMM Diag with 5-fold on z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMMDiag(PCA7, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM Diag model is", temp[c])
        minDCF5foldDiagPCA7.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldDiagPCA7, "GMM components", "PCA=7")



gmm = GMMDiag.GMMDiag()
minDCF5foldDiagPCA8 = []
print("Start GMM Diag with 5-fold on z normalized features PCA m=8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMMDiag(PCA8, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM Diag model is", temp[c])
        minDCF5foldDiagPCA8.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldDiagPCA8, "GMM components", "PCA=8")


gmm = GMMDiag.GMMDiag()
minDCF5foldDiagLDA = []
print("Start GMM Diag with 5-fold on z normalized features LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMMDiag(transformed_data2, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM Diag model is", temp[c])
        minDCF5foldDiagLDA.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldDiagLDA, "GMM components", "LDA")



gmm = GMMDiag.GMMDiag()
minDCF5foldPCALDA = []
print("Start GMM with 5-fold on z normalized features PCA+LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMM(transformed_data, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldPCALDA.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldPCALDA, "GMM components", "PCA+LDA")






#Tied Cov

plt.figure()
gmm = GMMTiedCov.GMMTiedCov()
minDCF5fold = []
print("Start GMM with 5-fold on z normalized features ")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMM(ZD, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5fold.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5fold, "GMM components", "Raw")



gmm = GMMTiedCov.GMMTiedCov()
minDCF5foldTiedPCA7 = []
print("Start GMM Tied-Cov with 5-fold on z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMMTied(PCA7, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM Tied-Cov model is", temp[c])
        minDCF5foldTiedPCA7.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldTiedPCA7, "GMM components", "PCA=7")



gmm = GMMTiedCov.GMMTiedCov()
minDCF5foldTiedPCA8 = []
print("Start GMM Tied-Cov with 5-fold on z normalized features PCA m=8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMMTied(PCA8, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM Tied-Cov model is", temp[c])
        minDCF5foldTiedPCA8.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldTiedPCA8, "GMM components", "PCA=8")

gmm = GMMTiedCov.GMMTiedCov()
minDCF5foldTiedLDA = []
print("Start GMM Tied-Cov with 5-fold on z normalized features LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMMTied(transformed_data2, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM Tied-Cov model is", temp[c])
        minDCF5foldTiedLDA.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldTiedLDA, "GMM components", "LDA")



gmm = GMMTiedCov.GMMTiedCov()
minDCF5foldPCALDA = []
print("Start GMM with 5-fold on z normalized features PCA+LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMM(transformed_data, L, gmm, numberOfSplitToPerform, prior=priors[i])
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldPCALDA.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldPCALDA, "GMM components", "PCA+LDA")



#single fold graphics
plt.figure()
#Full Cov
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA7, L)
gmm = GMM.GMM()
minDCF5foldPCA7 = []
print("Start GMM with single fold on z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    gmm.train(DTRSF, LTRSF, numberOfSplitToPerform)
    temp=metrics.minimum_detection_costs(gmm.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldPCA7.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldPCA7, "GMM components", "PCA=7")


(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA8, L)
gmm = GMM.GMM()
minDCF5foldPCA8 = []
print("Start GMM with single fold on z normalized features PCA m=8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    gmm.train(DTRSF, LTRSF, numberOfSplitToPerform)
    temp=metrics.minimum_detection_costs(gmm.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldPCA8.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldPCA8, "GMM components", "PCA=8")



(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(transformed_data2, L)
gmm = GMM.GMM()
minDCF5foldLDA = []
print("Start GMM with single fold on z normalized features LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    gmm.train(DTRSF, LTRSF, numberOfSplitToPerform)
    temp=metrics.minimum_detection_costs(gmm.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldLDA.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldLDA, "GMM components", "LDA")



plt.figure()
#Naive Bayes
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA7, L)
gmm = GMMDiag.GMMDiag()
minDCF5foldDiagPCA7 = []
print("Start GMM with single on z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    gmm.train(DTRSF, LTRSF, numberOfSplitToPerform)
    temp=metrics.minimum_detection_costs(gmm.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldDiagPCA7.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldDiagPCA7, "GMM components", "PCA=7")


(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA8, L)
gmm = GMMDiag.GMMDiag()
minDCF5foldDiagPCA8 = []
print("Start GMM with single fold on z normalized features PCA m=8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    gmm.train(DTRSF, LTRSF, numberOfSplitToPerform)
    temp=metrics.minimum_detection_costs(gmm.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldDiagPCA8.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldDiagPCA8, "GMM components", "PCA=8")





(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(transformed_data2, L)
gmm = GMMDiag.GMMDiag()
minDCF5foldDiagLDA = []
print("Start GMM with single fold on z normalized features LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    gmm.train(DTRSF, LTRSF, numberOfSplitToPerform)
    temp=metrics.minimum_detection_costs(gmm.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldDiagLDA.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldDiagLDA, "GMM components", "LDA")




plt.figure()
#Tied Cov
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA7, L)
gmm = GMMTiedCov.GMMTiedCov()
minDCF5foldTiedPCA7 = []
print("Start GMM with single fold on z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    gmm.train(DTRSF, LTRSF, numberOfSplitToPerform)
    temp=metrics.minimum_detection_costs(gmm.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldTiedPCA7.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldTiedPCA7, "GMM components", "PCA=7")


(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA8, L)
gmm = GMMTiedCov.GMMTiedCov()
minDCF5foldTiedPCA8 = []
print("Start GMM with single fold on z normalized features PCA m=8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    gmm.train(DTRSF, LTRSF, numberOfSplitToPerform)
    temp=metrics.minimum_detection_costs(gmm.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldTiedPCA8.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldTiedPCA8, "GMM components", "PCA=8")



(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(transformed_data2, L)
gmm = GMMTiedCov.GMMTiedCov()
minDCF5foldTiedLDA = []
print("Start GMM with single fold on z normalized features LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    gmm.train(DTRSF, LTRSF, numberOfSplitToPerform)
    temp=metrics.minimum_detection_costs(gmm.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
    for c in range(numberOfSplitToPerform):
        print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
        minDCF5foldTiedLDA.append(temp[c])
print("")
print("END")
MLlibrary.plotDCFGMM([2**(c+1) for c in range(numberOfSplitToPerform)], minDCF5foldTiedLDA, "GMM components", "LDA")





#single fold values


gmm = GMM.GMM()
minDCF5foldPCA7 = []
print("Start GMM with single fold on z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    temp = MLlibrary.fastKfoldGMM(PCA7, L, gmm, numberOfSplitToPerform, prior=priors[i])
    print("For", 2**(c+1), "components and prior=", priors[i], "the minDCF of the GMM model is", temp[c])
    minDCF5foldPCA7.append(temp[c])
print("")
print("END")


gmm = GMM.GMM()
print("Start GMM with 5-fold on z normalized features")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(ZD, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")


gmm = GMM.GMM()
print("Start GMM with 5-fold on z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(PCA7, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")

gmm = GMM.GMM()
print("Start GMM with 5-fold on z normalized features PCA m=8")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(PCA8, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")


gmm = GMM.GMM()
print("Start GMM with 5-fold on z normalized features LDA")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(transformed_data2, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")

gmm = GMM.GMM()
print("Start GMM with 5-fold on z normalized features PCA+LDA")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(transformed_data, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")



#Naive Bayes

gmm = GMMDiag.GMMDiag()
print("Start GMM with 5-fold on z normalized features")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(ZD, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")


gmm = GMMDiag.GMMDiag()
print("Start GMM with 5-fold on z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(PCA7, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")

gmm = GMMDiag.GMMDiag()
print("Start GMM with 5-fold on z normalized features PCA m=8")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(PCA8, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")


gmm = GMMDiag.GMMDiag()
print("Start GMM with 5-fold on z normalized features LDA")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(transformed_data2, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")

gmm = GMMDiag.GMMDiag()
print("Start GMM with 5-fold on z normalized features PCA+LDA")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(transformed_data, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")


#Tied Cov

gmm = GMMTiedCov.GMMTiedCov()
print("Start GMM with 5-fold on z normalized features")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(ZD, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")


gmm = GMMTiedCov.GMMTiedCov()
print("Start GMM with 5-fold on z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(PCA7, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")

gmm = GMMTiedCov.GMMTiedCov()
print("Start GMM with 5-fold on z normalized features PCA m=8")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(PCA8, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")


gmm = GMMTiedCov.GMMTiedCov()
print("Start GMM with 5-fold on z normalized features LDA")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(transformed_data2, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")

gmm = GMMTiedCov.GMMTiedCov()
print("Start GMM with 5-fold on z normalized features PCA+LDA")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(transformed_data, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")