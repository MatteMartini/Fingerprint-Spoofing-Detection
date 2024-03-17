import MLlibrary
import metrics
import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import PCA
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
    return U[:, ::-1][:, 0:m] #la prima parte serve per farli in ordine decrescente, perche l'operazione precedente ti da U ordinata dagli autovettori piu piccoli a quelli piu grandi!




if __name__ == '__main__':
 priors = [0.5]
D, L = MLlibrary.load('Train.txt')
ZD, mean, standardDeviation = MLlibrary.ZNormalization(D)
lambd = 1e-2
lr = LogisticRegression.LogisticRegression()
PCA7 = PCA.PCA(ZD, L, 7)
PCA8 = PCA.PCA(ZD, L, 8)
PCA9 = PCA.PCA(ZD, L, 9)

P2=LDA(ZD, L, 2)
transformed_data2=np.dot(P2.T, D)

PCA7 = PCA.PCA(ZD, L, 7)
P2L=LDA(PCA7, L, 2)
transformed_data=np.dot(P2L.T, PCA7)

lambdas=np.logspace(-4, 2.5, num=49)
plt.figure()



# SINGLE FOLD Z NORMALIZED LR
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(ZD, L)
minDCF = []
print("Start logistic regression on single fold of z normalized features")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        lr.train(DTRSF, LTRSF, l, 0.5) # pi_T = 0.5
        temp = metrics.minimum_detection_costs(lr.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        minDCF.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minDCF, "λ", "Raw")




# SINGLE FOLD Z NORMALIZED LR PCA WITH M=7
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA7, L)
minDCF7 = []
print("Start logistic regression on single fold of z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        lr.train(DTRSF, LTRSF, l, 0.5)
        temp=metrics.minimum_detection_costs(lr.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        minDCF7.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minDCF7, "λ", "PCA=7")




# SINGLE FOLD Z NORMALIZED LR PCA WITH M=8
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA8, L)
minDCF8 = []
print("Start logistic regression on single fold of z normalized features PCA m=8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        lr.train(DTRSF, LTRSF, l, 0.5)
        temp=metrics.minimum_detection_costs(lr.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        minDCF8.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minDCF8, "λ", "PCA=8")




# SINGLE FOLD Z NORMALIZED LR LDA WITH M=2
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(transformed_data2, L)
minLDA = []
print("Start logistic regression on single fold of z normalized features LDA m=2")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        lr.train(DTRSF, LTRSF, l, 0.5)
        temp=metrics.minimum_detection_costs(lr.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        minLDA.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minLDA, "λ", "LDA")


# SINGLE FOLD Z NORMALIZED LR LDA WITH M=2
(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(transformed_data, L)
minPCALDA = []
print("Start logistic regression on single fold of z normalized features PCA+LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        lr.train(DTRSF, LTRSF, l, 0.5)
        temp=metrics.minimum_detection_costs(lr.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        minPCALDA.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minPCALDA, "λ", "PCA+LDA")




# #Inizio nuovo grafico con K=5 folds
plt.figure()
MLlibrary.labels=[]

minDCF5foldRaw = []
print("Start logistic regression with 5-fold on z normalized features ")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        temp = MLlibrary.KfoldLR(ZD, L, lr, l, prior=priors[i])
        minDCF5foldRaw.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minDCF5foldRaw, "λ", "Raw")



minDCF5fold7 = []
print("Start logistic regression with 5-fold on z normalized features PCA m=7")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        temp = MLlibrary.KfoldLR(PCA7, L, lr, l, prior=priors[i])
        minDCF5fold7.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minDCF5fold7, "λ", "PCA=7")





minDCF5fold8 = []
print("Start logistic regression with 5-fold on z normalized features PCA m=8")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        temp = MLlibrary.KfoldLR(PCA8, L, lr, l, prior=priors[i])
        minDCF5fold8.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minDCF5fold8, "λ", "PCA=8")





minDCF5foldLDA = []
print("Start logistic regression with 5-fold on z normalized features LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        temp = MLlibrary.KfoldLR(transformed_data2, L, lr, l, prior=priors[i])
        minDCF5foldLDA.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minDCF5foldLDA, "λ", "LDA")



minDCF5foldPCALDA = []
print("Start logistic regression with 5-fold on z normalized features PCA+LDA")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        temp = MLlibrary.KfoldLR(transformed_data, L, lr, l, prior=priors[i])
        minDCF5foldPCALDA.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minDCF5foldPCALDA, "λ", "PCA+LDA")