import random
import pickle
import json
import nltk
import torch
import numpy as np
import torch.nn as nn

#------------------------------------------------------------------1
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('Intents.json').read())

#------------------------------------------------------------------2

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
words = []
classes = []
documents = []
ignore_letters = [',', '.', '?', '!' ]

for intent in intents['intents']:
  for pattern in intent['patterns']:
    word_list = nltk.word_tokenize(pattern)
    words.append(word_list)
    documents.append((word_list, intent['tag']))
    if intent['tag'] not in  classes:
      classes.append(intent['tag'])

# print(documents)
# print(words)

#------------------------------------------------------------------3

from nltk.stem import WordNetLemmatizer

ignore_letters = [',', '.', '?', '!' ] #  defined the 'ignore_letters' list

lemmatizer = WordNetLemmatizer()

# words = [['Hello'], ['Hi'], ['Hey'], ['Good', 'morning'], ['Good', 'afternoon'], ['Good', 'evening'], ['How', 'are', 'you', '?'], ['What', "'s", 'up', '?'], ['How', "'s", 'it', 'going', '?'], ['Tell', 'me', 'about', 'yourself', '.'], ['Who', 'are', 'you', '?'], ['What', 'do', 'you', 'do', '?'], ['What', 'is', 'your', 'purpose', '?'], ['Introduce', 'yourself', '.'], ['What', 'is', 'your', 'name', '?'], ['Who', 'are', 'you', '?'], ['Can', 'you', 'tell', 'me', 'your', 'name', '?'], ['May', 'I', 'know', 'your', 'name', '?'], ['What', 'should', 'I', 'call', 'you', '?'], ['How', 'old', 'are', 'you', '?'], ['What', 'is', 'your', 'age', '?'], ['When', 'were', 'you', 'born', '?'], ['Are', 'you', 'young', 'or', 'old', '?'], ['What', "'s", 'your', 'birthdate', '?'], ['What', 'time', 'is', 'it', '?'], ['Do', 'you', 'have', 'the', 'time', '?'], ['Can', 'you', 'tell', 'me', 'the', 'current', 'time', '?'], ['What', "'s", 'the', 'time', 'right', 'now', '?'], ['Do', 'you', 'know', 'what', 'time', 'it', 'is', '?'], ['What', 'do', 'you', 'do', 'for', 'a', 'living', '?'], ['Tell', 'me', 'about', 'your', 'work', '.'], ['What', 'is', 'your', 'profession', '?'], ['What', 'is', 'your', 'job', '?'], ['How', 'do', 'you', 'spend', 'your', 'time', '?'], ['Tell', 'me', 'a', 'joke', '.'], ['Do', 'you', 'know', 'any', 'jokes', '?'], ['Can', 'you', 'share', 'a', 'funny', 'joke', '?'], ['I', 'could', 'use', 'a', 'laugh', '.', 'Tell', 'me', 'something', 'funny', '.'], ['What', "'s", 'your', 'favorite', 'joke', '?'], ['What', 'are', 'your', 'hobbies', '?'], ['Do', 'you', 'have', 'any', 'interests', '?'], ['Tell', 'me', 'about', 'your', 'hobbies', '.'], ['What', 'do', 'you', 'like', 'to', 'do', 'for', 'fun', '?'], ['What', 'are', 'your', 'favorite', 'activities', '?']]  # Replace this with your actual list of words

# Lemmatize the words and filter out words containing ignore_letters
words = [lemmatizer.lemmatize(word) for sublist in words for word in sublist if not any(letter in word for letter in ignore_letters)]

words = sorted(set(words))
classes = sorted(set(classes))
# print(words)
# print(classes)

#-------------------------------------------------------------------4

pickle.dump(words, open('words.pkl', 'wb')) # Write binary mode
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []

output_empty = [0] * len(classes)


#-------------------------------------------------------------------5

for document in documents:
  bag = []
  word_patterns = document[0]
  word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
  for word in words:
    bag.append(1) if word in word_patterns else bag.append(0)

  output_row = list(output_empty)
  output_row[classes.index(document[1])] = 1
  training.append([bag, output_row])
#----------------------------------------------------------------------6
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

#----------------------------------------------------------------------6

from keras.backend import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD

model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose= 1)

model.save('trained_model.h5', hist)

print("Congratulations! Almost done")

#-----------------------------------------------------------------------7
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('Intents.json').read())

words  = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('trained_model.h5')

def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

  return sentence_words

def bag_of_words(sentence):
  sentence_words = clean_up_sentence(sentence)
  bag = [0] * len(words)

  for w in sentence_words:
    for i, word in enumerate(words):
      if word  == w:
        bag[i] = 1

  return np.array(bag)

def predict_class(sentence):
  bow = bag_of_words(sentence)
  res = model.predict(np.array([bow]))[0]

  ERROR_THRESHOLD = 0.25
  results = [[i, r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

  results.sort(key = lambda x : x[1], reverse = True)
  return_list = []

  for r in results:
    return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

  return return_list

def get_responses(intents_list, intents_json):
  tag = intents_list[0]['intent']
  # print(tag)
  list_of_intents = intents_json['intents']
  result = "I don't understand, Say Again!"  # Assign an initial value to 'result'

  for i in list_of_intents:
    if i['tag'] == tag:
      result = random.choice(i['responses'])
      break

  # print(tag)
  return result
  
#---------------------------->final step<-------------------------------
print("\n 👇Ready! Juvi Bot is active👇\n")

while True:
  message = input()
  ints = predict_class(message)
  res = get_responses(ints, intents)
  print("Juvi Bot :👉", res)
  if message == "quit" or "shutdown":
    print("Thanks for use me!")
    break