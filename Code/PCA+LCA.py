
import numpy
import matplotlib
import matplotlib.pyplot as plt  #quando vedrai plt vorrà dire pyplot
import scipy

#come previsto, usiamo una matrice in cui ogni COLONNA contiente un sample, mentre sulle righe metti gli attributi!!
#la classe è gia nel csv come ultimo campo, devi usare un vettore in cui le carichi tutte in ordine!
def vcol(v):
    return v.reshape((v.size, 1))
def vrow(v):
    return v.reshape((1, v.size))

def load(fname):
    DList = []
    labelsList = []
    # hLabels = {  #dizionario in cui metti in corrispondenza il nome della classe col suo valore!
    #     'Spoofed-fingerprint': 0,
    #     'Authentic-fingerprint': 1
    #     }

    with open(fname) as f:
        for line in f: #modo per iterare se non conosci il numero di righe!
            try:
                attrs = line.split(',')[0:10]
                attrs = vcol(numpy.array([float(i) for i in attrs])) #ATT! metodo per creare un numpy array avendo i valori in un vettore, col cast a float
                label = line.split(',')[-1].strip() #con questo ricavi dal nome della classe presa dal file il valore corrispondente salvato nel dizionario, e aggiungi la classe assegnata al vettore in cui stai mettendo tutte le etichette assegnate in ordine
                DList.append(attrs) #in Dlist metti i primi 4 campi, cioe dei vettori contenti i valori per ogni attributo di un sample. Cosi però è un vettore di vettori, non è una matrice 4x150 che vogliamo noi di numpy! Dovrai usare hstack
                labelsList.append(label) #qui avrai l'elenco delle etichette assegnate in ordine dalla 1 alla 150
            except:
                pass #cosi se c'è qualche errore sintattico va avanti e non da errore!

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)  # i valori degli attributi in Dlist li trasformi nella matrice 4x150, mentre per l'elenco delle etichette le metti in un array e basta!


def load2(): #corciatoia, in cui pero devi trasporre la matrice perche ti da una matrice 150x4

    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def plot_hist(D, L):

    D0 = D[:, L==0] #stai facendo un filtro booleano usando L==0, dicendogli che in D0 metti tutte le iris della classe 0, ovvero Iris-setosa. qUINDI USCIRAANO 3 MATRICI 4X50!! Essendoci 50 sample per ogni classe
    D1 = D[:, L==1] #qui metti tutte le iris della classe 1!



    for dIdx in range(10): #cicla sui 4 possibili attributi! E per ogni attributo plotti 3 istogrammi, uno per ogni classe
        plt.figure()  #ATT! Fai una sola figure per tutti e 3 gli istogrammi! Saranno tutti sullo stesso grafico ma sovrapposti!
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Spoofed-fingerprint')  #devi dare dei numpy array MONODIMENSIONALI a hist, come fai con D0[dIdx, :]
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Authentic-fingerprint')
        #bins sono il numero di colonnine che vuoi sul grafico, density = true normalizza l'istogramma dandoti le densità di probabilità, alpha ci dice da(0 a 1) quanto vuoi acceso il colore che usi per le colonne, la label è quella per l'asse y
        plt.legend() #ti crea una legenda che associa il colore dell'istogramma alla classe che gli corrisponde (lo prende dalla label dell'asse y il nome della classe)
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.jpg' % dIdx) #CON QUESTA RIGA CREI UN PDF, MA ESSENDO NEL FOR LO FA PER OGNI GRAFICO!!!
    plt.show() #legato a figure, e blocca l'esecuzione ficnhe non chiudi la nuova finestra aperta col grafico!

def plot_scatter(D, L):
    
    D0 = D[:, L==0] 
    D1 = D[:, L==1]

#con lo scatter invece fai il confronto a 2 a 2 tra i vari attributi
    for dIdx1 in range(10):
        for dIdx2 in range(10):
            if dIdx1 == dIdx2: #cosi escludi di plottare il confronto di un attributo con se stesso
                continue
            plt.figure()
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Spoofed-fingerprint')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Authentic-fingerprint')
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('scatter_%d_%d.jpg' % (dIdx1, dIdx2))
        plt.show()

def plot_hist2(D, L):  #cambiata

    D0 = D[:, L==0] #stai facendo un filtro booleano usando L==0, dicendogli che in D0 metti tutte le iris della classe 0, ovvero Iris-setosa. qUINDI USCIRAANO 3 MATRICI 4X50!! Essendoci 50 sample per ogni classe
    D1 = D[:, L==1] #qui metti tutte le iris della classe 1!



    plt.figure()  #ATT! Fai una sola figure per tutti e 3 gli istogrammi! Saranno tutti sullo stesso grafico ma sovrapposti!
    plt.hist(D0[0],  bins = 10, density = True, alpha = 0.4, label = 'Spoofed-fingerprint')  #devi dare dei numpy array MONODIMENSIONALI a hist, come fai con D0[dIdx, :]
    plt.hist(D1[0],  bins = 10, density = True, alpha = 0.4, label = 'Authentic-fingerprint')
    #bins sono il numero di colonnine che vuoi sul grafico, density = true normalizza l'istogramma dandoti le densità di probabilità, alpha ci dice da(0 a 1) quanto vuoi acceso il colore che usi per le colonne, la label è quella per l'asse y
    plt.legend() #ti crea una legenda che associa il colore dell'istogramma alla classe che gli corrisponde (lo prende dalla label dell'asse y il nome della classe)
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig('hist_%d.png' % 0) #CON QUESTA RIGA CREI UN PDF, MA ESSENDO NEL FOR LO FA PER OGNI GRAFICO!!!
    plt.show() #legato a figure, e blocca l'esecuzione ficnhe non chiudi la nuova finestra aperta col grafico!


def plot_scatter2(D, L, title=''):    
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]    
   
    plt.scatter(D1[0], (-1)*D1[1], label='Authentic-fingerprint')
    plt.scatter(D0[0], (-1)*D0[1], label='Spoofed-fingerprint')
    plt.legend()
    plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
    plt.title(title)
    plt.savefig('scat_%s.png' % title)
    plt.show()

#PCA and LDA - Lab 3  m glielo passi come parametro, è il numero di autovettori da prnedere
def PCA(D, m):  #Per il calcolo di P qui si è usato il secondo metodo scritto sul pdf, ovvero quello che ti da gli autovettori gia in ordine decrescente!
    mu = vcol(D.mean(1))
    C = numpy.dot(D - mu, (D - mu).T) / D.shape[1]   #D - mu è Dc, cioe la matrice dei dati centrata! .T ti da la trasposta!! é proprio la formula del prodotto scritta in python. D.shape[1] è N della formula, ovvero 150
    U, _, _ = numpy.linalg.svd(C)  #è il comando per la Singolar Value Decomposition che ti da già la U con eigencìvectors ordinati dal piu alto al piu basso! Nella parte a sinistra ci sono i trattini bassi per dire che quelle robe non ti servono, quindi non serve/sprechi delle variabili per esso!
    P = U[:, 0:m]

    return P


def SbSw(D, L):
    SB = 0
    SW = 0
    mu = vcol(D.mean(1))  #media di tutti i sample di una riga per ogni attributo, è un vettore di 4 elementi in questo caso, perche ci sono 4 possibili attributi => è la media del dataset, cioe la media di tutti i sample, distinto per ogni attributo!
    for i in range(L.max() + 1): #L.max() +1 ti da il numero di classi differenti tra i dati passati
        DCls = D[:, L == i]  #ti prendi cosi tutti i sample della classe in analisi, e saranno ad classe 0, poi classe 1 ecc.
        muCls = vcol(DCls.mean(1)) #media degli elementi di una classe! Grazie alla riga prima escludi gli elementi della classe in analisi
        SW += numpy.dot(DCls - muCls, (DCls - muCls).T)  #ad ogni iterazione aggiungi il contributo della parte a destra
        SB += DCls.shape[1] * numpy.dot(muCls - mu, (muCls - mu).T)  #DCls.shape[1] corrisponde al nc della formula

    SW /= D.shape[1] # è il fratto N che sta nelle 2 formule
    SB /= D.shape[1]
    return SB, SW

def LDA(D, L, m):
    SB, SW = SbSw(D, L)
    s, U = scipy.linalg.eigh(SB, SW) 
    return U[:, ::-1][:, 0:m] #la prima parte serve per farli in ordine decrescente, perche l'operazione precedente ti da U ordinata dagli autovettori piu piccoli a quelli piu grandi!


if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load('Train.txt') #glielo passi come parametro il nome del file, non con argv. Con questa funzione carichi i dati nella matrice 4x150 in D, e le 150 etichette nel vettore L


    #Statistiche implementate da te!
    mu = 0
    mu = D.mean(1) #fai la media lungo le colonne, ottenendo cosi un vettore 1-D di 4 elementi ! 
    DC = D - vcol(mu)  #funziona grazie al broadcasting!!
    
    
    m=2
    P1=PCA(D, m)
    transformed_data=numpy.dot(P1.T, D)
   # P2=LDA(D, L, m)
    P2=LDA(transformed_data, L, m)
    transformed_data2=numpy.dot(P2.T, transformed_data)
    plot_scatter2(transformed_data2, L, '')
   # transformed_data=numpy.dot(P1.T, D)
    #plot_scatter2(transformed_data, L, 'PCA+LDA')
    
    #transformed_data2=numpy.dot(P2.T, transformed_data)
   # plot_scatter2(transformed_data2, L, 'PCA+LDA')
   # plot_hist2(transformed_data2, L)