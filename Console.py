import numpy as np
from sentence_transformers import SentenceTransformer
import re
from nltk.corpus import stopwords
import nltk

# Stopwords herunterladen
nltk.download('stopwords')

# Stopwords setzen
stop_words = set(stopwords.words('english'))
transformer = 'nreimers/albert-small-v2'

# Textbereinigung
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # URLs entfernen
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Nicht-Alphabetische Zeichen entfernen
    text = text.lower()  # Kleinbuchstaben
    text = text.split()  # In Wörter teilen
    text = [word for word in text if word not in stop_words]  # Stopwörter entfernen
    return text

# Laden des trainierten Modells und der Daten
print("Lade Modell...")
loaded_model = np.load('neuronal_network.npz')
size = int(len(loaded_model)/2)+1
weight = [loaded_model[f'w{i}'] for i in range(size-1)]
bias = [loaded_model[f'b{i}'] for i in range(size-1)]
bert_model = SentenceTransformer(transformer)

emotions = ["neutral", "worry", "happiness", "sadness", "love", "hate"]

# Funktionen für Vorhersage und Textkodierung
def sigmoid(value): 
    return 1 / (1 + np.exp(-value))

def forward(bias, weight, x): 
    pre = bias + np.dot(weight, x)
    return sigmoid(pre)

def encode_text(text):
    return bert_model.encode([text])

def predict_hate_speech(sentence):
    encoded_sentence = encode_text(sentence)
    l = [0] * size
    for i in range(1, size):
        if i == 1:
            l[1] = forward(bias[0], weight[0], encoded_sentence.T)
        else:
            l[i] = forward(bias[i-1], weight[i-1], l[i-1])

    prediction = np.argmax(l[size-1])
    return emotions[prediction]

# Hauptprogramm: Wiederholte Eingabe bis "exit"
if __name__ == "__main__":
    print('Gib einen Satz ein, um das Sentiment zu klassifizieren (oder "exit" zum Beenden):')

    while True:
        user_input = input(">> ")

        if user_input.lower() == "exit":
            print("Programm beendet.")
            break
        
        if user_input.strip() == "":
            print("Fehler: Bitte gib einen gültigen Text ein!")
        else:
            prediction = predict_hate_speech(user_input)
            print(f"Das vorhergesagte Sentiment ist: {prediction}")
