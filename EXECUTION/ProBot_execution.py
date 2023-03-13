import numpy as np
import nltk
import random
import json
import pickle
from datetime import date
import joblib
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('ProBot_DataSet.json').read())
data_details = json.loads(open('ProBot_DetailSet.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = joblib.load('chatbot_model.pkl')

date_var = date.today().strftime("%d")
date_var = int(date_var)
if date_var < 10:
  todays_date = 9
else:
  todays_date = 10

if todays_date==9:
  list_of_details = data_details['intents_9']
else:
  list_of_details = data_details['intents_10']


def value_adder(result,tag):
  for de in list_of_details:
    if tag == de['tag']:
      if tag+"_start" in result:
        start_time = de['time_start']
        result = result.replace("["+tag+"_start]",str(start_time))
      if tag+"_end" in result:
        end_time = de['time_end']
        result = result.replace("["+tag+"_end]",str(end_time))
      if tag+"_name" in result:
        venue_name = de['venue_name']
        result = result.replace("["+tag+"_name]",str(venue_name))
      break
  return result

def sentence_cleaning(sentence):
  sen_words = nltk.word_tokenize(sentence)
  sen_words = [lemmatizer.lemmatize(word) for word in sen_words]
  return sen_words

def wordBag(sentence):
  sen_words = sentence_cleaning(sentence)
  bag = [0] * len(words)
  for w in sen_words:
    for i,word in enumerate(words):
      if word == w:
        bag[i] = 1
  return np.array(bag)

def predict_class(sentence):
  bow = wordBag(sentence)
  res = model.predict(np.array([bow]))[0]
  ERROR_THR = 0.25
  results = [[i, r] for i, r in enumerate(res) ]
  results.sort(key=lambda x:x[1], reverse=True)
  return_list=[]
  for r in results:
    return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
  return return_list

def get_response(intents_list, intents_json):
  max_prob_intent = intents_list[0]['intent']
  max_prob = float(intents_list[0]['probability'])
  if max_prob < 0.8:
    return "I'm sorry, I didn't understand your question."
  for intent in intents_list:
    if float(intent['probability']) > max_prob:
      max_prob = float(intent['probability'])
      max_prob_intent = intent['intent']
  tag = max_prob_intent
  list_of_intents = intents_json['intents']
  for i in list_of_intents:
    if i['tag'] == tag:
      result = random.choice(i['responses'])
      if '[' and ']' in result:
        result = value_adder(result,tag)
      break
  return result




def process_begin(text):
    flag = True
    while (flag == True):
      user_response = str(text)
      print("USER:",user_response)
      user_response = user_response.lower()
      if(user_response != 'bye'):
        if(user_response == 'thank you' or user_response == 'thanks'):
          flag = False
          return "You are Welcome.."
        else:
          ints = predict_class(user_response)
          res = get_response(ints, intents)
          return res
      else:
          flag = False
          return "GoodBye!"



app = Flask(__name__,template_folder="templates")

@app.route("/")
def aut():
    return render_template("ProBot_Frontend.html")

@app.route("/receive-data", methods=["POST"])
def receive_data():
    data = request.get_json()
    print("data:",data)
    result = process_begin(str(data))
    response = result
    return str(response)

if __name__ == "__main__":
    app.run()