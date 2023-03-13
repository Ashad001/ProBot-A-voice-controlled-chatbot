import numpy as np
import nltk
import pandas as pd
import random
import json
import pickle
import random
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

nltk.download ('punkt')
nltk.download ('wordnet')
nltk.download ('omw-1.4')

lemmatizer = WordNetLemmatizer()
raw_doc = json.loads(open('ProBot_DataSet.json').read())

words = []
classes = []
documents = []
ignore_let = ['?', '!', ",", "."]

for intent in raw_doc['intents']:
  for pat in intent['patterns']:
    word_list = nltk.word_tokenize(pat)
    words.extend(word_list)
    documents.append((word_list, intent['tag']))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_let]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

train = []
output_emp = [0] * len(classes)
for doc in documents:
  bag = []
  word_pat = doc[0]
  word_pat = [lemmatizer.lemmatize(word.lower()) for word in word_pat]
  for word in words:
    bag.append(1) if word in word_pat else bag.append(0)
  out_row = list(output_emp)
  out_row[classes.index(doc[1])] = 1
  train.append([bag, out_row])

random.shuffle(train)
train = np.array(train, dtype=object)

train_x = list(train[:, 0])
train_y = list(train[:, 1])
t_x, test_x, t_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, activation='relu', solver='adam', random_state=42)
model.fit(train_x, train_y)

joblib.dump(model, 'chatbot_model.pkl')
print("Done")