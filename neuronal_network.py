import numpy as np
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


#Größe des Netzwerkes festlegen (size = Anzahl der hiddenlayer + Input und Output)

gr = [768,50,20,6]
size = len(gr)
transformer = 'nreimers/albert-small-v2' #wurde offline genommen!

#Weigths und Biases festlegen: Weights zufällig, Biases auf 0
weight=[0]*size
bias=[0]*size
for i in range(size-1):
    weight[i] = np.random.uniform(-0.5,0.5,(gr[i+1],gr[i]))
    bias[i] = np.zeros((gr[i+1],1))
    

#Methoden definieren
def sigmoid(value): 
    return 1/ (1 + np.exp(-value))

def forward(bias, weight, layer): 
    pre = bias + weight @ layer
    return(sigmoid(pre))

def forward_propagation(bias, weight, input):
    l=[0]*size
    for i in range(1, size):
        if i == 1:
            l[1] = forward(bias[0], weight[0], input)
        else:
            l[i] = forward(bias[i-1], weight[i-1], l[i-1])
    return l

def test(sentence, label):
    correct = 0
    count = 0
    for sentence, label in zip(sentence, label):
        sentence.shape+=(1,)
        label.shape+=(1,)
        l = forward_propagation(bias=bias, weight=weight, input=sentence)

        correct += int(np.argmax(l[size-1]) == np.argmax(label))
        count +=1

    return round((correct/count)*100,2)

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


#import Stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#Daten einlesen
df = pd.read_csv(f"https://raw.githubusercontent.com/Luismkl/Neuronal-Network/refs/heads/main/tweet_emotions.csv")

emotions = ["neutral", "worry", "happiness", "sadness", "love", "hate"]
df = df[df.sentiment.isin(emotions)]


target_count = df['sentiment'].value_counts()['sadness']
balanced_df = pd.concat([
    df[df['sentiment'] == 'worry'].sample(target_count, random_state=42),
    df[df['sentiment'] == 'neutral'].sample(target_count, random_state=42),
    df[df['sentiment'] == 'happiness'].sample(target_count, random_state=42),
    df[df['sentiment'] == 'sadness'],
    df[df['sentiment'] == 'love'],
    df[df['sentiment'] == 'hate']
])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

df_test, df_train = train_test_split(balanced_df , test_size=0.20, random_state=42)
df_vali, df_train = train_test_split(df_train , test_size=0.25, random_state=42)

#Stopwörter entfernen
#df_train['content'] = df_train['content'].apply(remove_stopwords)
#df_test['content'] = df_test['content'].apply(remove_stopwords)


#BERT modell zum sätze einlesen
class BertModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SentenceTransformer(transformer, trust_remote_code=True)
        return cls._instance

model = BertModelSingleton.get_instance()
sentences = model.encode(df_train["content"].tolist())
sentence_test = model.encode(df_test["content"].tolist())


#Labels einlesen
labels=np.empty((0,6))
for i in df_train['sentiment'].values:
    if i == "neutral":
        newrow = np.array([[1,0,0,0,0,0]])
    elif i == "worry":
        newrow = np.array([[0,1,0,0,0,0]])
    elif i == "happiness":
        newrow = np.array([[0,0,1,0,0,0]])
    elif i == "sadness":
        newrow = np.array([[0,0,0,1,0,0]])
    elif i == "love":
        newrow = np.array([[0,0,0,0,1,0]])
    elif i == "hate":
        newrow = np.array([[0,0,0,0,0,1]])
    else:
        print("ERROR" + i)
    labels = np.insert(labels, len(labels), newrow, axis=0)


#LAbels für Test einlesen
labels_test=np.empty((0,6))
for i in df_test['sentiment'].values:
    if i == "neutral":
        newrow = np.array([[1,0,0,0,0,0]])
    elif i == "worry":
        newrow = np.array([[0,1,0,0,0,0]])
    elif i == "happiness":
        newrow = np.array([[0,0,1,0,0,0]])
    elif i == "sadness":
        newrow = np.array([[0,0,0,1,0,0]])
    elif i == "love":
        newrow = np.array([[0,0,0,0,1,0]])
    elif i == "hate":
        newrow = np.array([[0,0,0,0,0,1]])
    else:
        print("ERROR" + i)
    labels_test = np.insert(labels_test, len(labels_test), newrow, axis=0)




#-------START NN-------

correct = 0 #Anzahl korrekte Ergebnisse
count = 0 #Anzahl Durchläufe pro Epoche bzw. Testgröße

epoch = 10 #Anzahl der Epochen
learnrate = 0.005
batch_size = 30 #Wie groß ist eine Untergruppe des Testdatensatzes

for epoche in range(epoch):
    w_change=[0]*size
    b_change=[0]*size

    for i in range(0, len(sentences), batch_size):

        batch = sentences[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        
        for sentence,label in zip(batch,batch_labels):

            sentence.shape+=(1,)
            label.shape+=(1,)

            l = forward_propagation(bias=bias, weight=weight, input=sentence)

            #Error-wert berechnen
            err = 1/len(l[size-1]) * np.sum((l[size-1] - label)**2, axis=0)
            correct += int(np.argmax(l[size-1]) == np.argmax(label))
            count+=1
            
            #Backpropagation Start
            
            #Zwischenspeichern der Weights
            w_final = [i for i in weight]
    
            delta_l = [0]*size
            for i in range (size-1, 0, -1):
                if i == size-1:
                    delta_l[i] = l[size-1] - label
                else:
                    #Derivative berechnen: Backpropagation-weights * Ableitung von Sigmoid
                    delta_l[i] = np.transpose(w_final[i]) @ delta_l[i+1] * (l[i]*(1-l[i]))
                if i == 1:
                    w_change[i-1] += -learnrate* delta_l[i] @ np.transpose(sentence)
                else:
                    #Vorüberhende Speicherung der Weight-Änderungen in w_change bevor nach Epoche Durchschnitt genommen wird
                    w_final[i-1] += -learnrate* delta_l[i] @ np.transpose(l[i-1])
                    w_change[i-1] += -learnrate* delta_l[i] @ np.transpose(l[i-1])

                b_change[i-1] += -learnrate* delta_l[i]


        #Updaten der Weight/ Biases nach Abschluss einer Epoche: Durchschnitt nehmen
        for i in range(size):
            weight[i] += w_change[i] / count
            bias[i] += b_change[i] / count
 

    #Ausgeben der Genauigkeit nach jeder Iteration    
    print(f"Acc: {round((correct/count)*100,2)} %")
    correct=0
    count=0


#Testen des Modells
acc_test=test(sentence_test, labels_test)
print("")
print(f"Test: {acc_test} %")


#trainiertes Modell speichern
dict={}
for i in range(size-1):
    dict[f'w{i}']=weight[i]
    dict[f'b{i}']=bias[i]
np.savez('neuronal_network.npz', **dict)

