import numpy as np
import scipy.optimize
import LogRegFunctions
import LogisticRegression
import MLlibrary
import metrics
import matplotlib.pyplot as plt
import PCA

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

P2=LDA(ZD, L, 2)
transformed_data2=np.dot(P2.T, D)

PCA7 = PCA.PCA(ZD, L, 7)
P2L=LDA(PCA7, L, 2)
transformed_data=np.dot(P2L.T, PCA7)


print("Start logistic regression with 5-fold on z normalized features")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        temp = MLlibrary.KfoldLR(ZD, L, lr, lambd, prior=priors[i], pi_T=priors[j])
        print("minDCF for Log-Reg (lambda=10^(-3), pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")



print("Start logistic regression with 5-fold on z normalized features PCA m=7")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        temp = MLlibrary.KfoldLR(PCA7, L, lr, lambd, prior=priors[i], pi_T=priors[j])
        print("minDCF for Log-Reg (lambda=10^(-3), pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")




print("Start logistic regression with 5-fold on z normalized features PCA m=8")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        temp = MLlibrary.KfoldLR(PCA8, L, lr, lambd, prior=priors[i], pi_T=priors[j])
        print("minDCF for Log-Reg (lambda=10^(-3), pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")


print("Start logistic regression with 5-fold on z normalized features LDA m=2")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        temp = MLlibrary.KfoldLR(transformed_data2, L, lr, lambd, prior=priors[i], pi_T=priors[j])
        print("minDCF for Log-Reg (lambda=10^(-3), pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")


print("Start logistic regression with 5-fold on z normalized features PCA+LDA")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        temp = MLlibrary.KfoldLR(transformed_data, L, lr, lambd, prior=priors[i], pi_T=priors[j])
        print("minDCF for Log-Reg (lambda=10^(-3), pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")


# single fold

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(ZD, L)
print("Start logistic regression with single fold on z normalized features")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        lr.train(DTRSF, LTRSF, lambd, 0.5)
        temp=metrics.minimum_detection_costs(lr.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        print("minDCF for Log-Reg (lambda=10^(-3), pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA7, L)
print("Start logistic regression with single fold on z normalized features PCA m=7")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        lr.train(DTRSF, LTRSF, lambd, 0.5)
        temp=metrics.minimum_detection_costs(lr.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        print("minDCF for Log-Reg (lambda=10^(-3), pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(PCA8, L)
print("Start logistic regression with single fold on z normalized features PCA m=8")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        lr.train(DTRSF, LTRSF, lambd, 0.5)
        temp=metrics.minimum_detection_costs(lr.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        print("minDCF for Log-Reg (lambda=10^(-3), pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")


(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(transformed_data2, L)
print("Start logistic regression with single fold on z normalized features LDA")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        lr.train(DTRSF, LTRSF, lambd, 0.5)
        temp=metrics.minimum_detection_costs(lr.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        print("minDCF for Log-Reg (lambda=10^(-3), pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")

(DTRSF, LTRSF), (DEVSF, LEVSF) = MLlibrary.split_db_singleFold(transformed_data, L)
print("Start logistic regression with single fold on z normalized features PCA+LDA")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        lr.train(DTRSF, LTRSF, lambd, 0.5)
        temp=metrics.minimum_detection_costs(lr.predictAndGetScores(DEVSF), LEVSF, priors[i], 1, 10)
        print("minDCF for Log-Reg (lambda=10^(-3), pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")
