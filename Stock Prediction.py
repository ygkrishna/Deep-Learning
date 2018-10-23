# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:19:07 2018

@author: Gopi
"""

import pandas as pd
import numpy as np
import statistics
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import initializers
from keras.layers import Dropout, Activation, Embedding, Convolution1D, MaxPooling1D, Input, Dense, merge
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.layers import concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers



##Import created contractions as a dictionary
import sys
sys.path.append('XXXXXX/Stock Prediction')
import dictionary
###Read data
dj = pd.read_csv("XXXXXX/DJIA_table.csv")
news = pd.read_csv("XXXXXX/RedditNews.csv")

# Compare the number of unique dates. We want matching values.
print(len(set(dj.Date)))
print(len(set(news.Date)))
# Remove the extra dates that are in news
news = news[news.Date.isin(dj.Date)]
print(len(set(dj.Date)))
print(len(set(news.Date)))
'''
# Calculate the difference in opening prices between the following and current day.
# The model will try to predict how much the Open value will change beased on the news.
dj = dj.set_index('Date').diff(periods=1)
dj['Date'] = dj.index
dj = dj.reset_index(drop=True)
# Remove unneeded features
dj = dj.drop(['High','Low','Close','Volume','Adj Close'], 1)

# Remove top row since it has a null value.
dj = dj[dj.Open.notnull()]'''

##Create Mid Values
#dj['Mid'] = (dj.High+dj.Low)/2.0
# Remove unneeded features
dj = dj.drop(['High','Low','Close','Volume','Adj Close'], 1)

# Create a list of the opening prices and their corresponding daily headlines from the news
price = []
headlines = []
for row in dj.iterrows():
    daily_headlines = []
    date = row[1]['Date']
    price.append(row[1]['Open'])
    for row_ in news[news.Date==date].iterrows():
        daily_headlines.append(row_[1]['News'])
    
    # Track progress
    headlines.append(daily_headlines)
    if len(price) % 500 == 0:
        print(len(price))

# Compare lengths to ensure they are the same
print(len(price))
print(len(headlines))

# Compare the number of headlines for each day
print(max(len(i) for i in headlines))
print(min(len(i) for i in headlines))
print(statistics.mean(len(i) for i in headlines))

'''Remove unwanted characters and format the text to create fewer nulls word embeddings'''
def clean_text(text, remove_stopwords = True):
    # Convert words to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in dictionary.contractions:
                new_text.append(dictionary.contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Remove Numbers
    text = ''.join([i for i in text if not i.isdigit()])
    # Format words, numbers and remove unwanted characters
    tokenizer = RegexpTokenizer(r'\w+')	
    text = " ".join(tokenizer.tokenize(text))
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    text = " ".join([i for i in text.split() if i not in stop_words])
    # stemming of words
    #porter = PorterStemmer()
    #text = ' '.join(porter.stem(token) for token in nltk.word_tokenize(text))
    # Lemmitization
    # lemmatize
    lmtzr = WordNetLemmatizer()
    text = ' '.join(lmtzr.lemmatize(token) for token in nltk.word_tokenize(text))
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'0,0', '00', text) 
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)
    
    return text

# Clean the headlines
clean_headlines = []

for daily_headlines in headlines:
    clean_daily_headlines = []
    for headline in daily_headlines:
        clean_daily_headlines.append(clean_text(headline))
    clean_headlines.append(clean_daily_headlines)
    
# Take a look at some headlines to ensure everything was cleaned well
clean_headlines[0]

# Find the number of times each word was used and the size of the vocabulary
word_counts = {}

for data in clean_headlines:
    for headline in data:
        for word in headline.split():
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1
            
print("Size of Vocabulary:", len(word_counts))

# Load GloVe's embeddings
embeddings_index = {}
with open('XXXXXX/glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))
# Find the number of words that are missing from GloVe, and are used more than our threshold.
missing_words = 0
threshold = 10

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1
            
missing_ratio = round(missing_words/len(word_counts),4)*100
            
print("Number of words missing from GloVe:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

# Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe
#dictionary to convert words to integers
vocab_to_int = {} 

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)
    
# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total Number of Unique Words:", len(word_counts))
print("Number of Words we will use:", len(vocab_to_int))
print("Percent of Words we will use: {}%".format(usage_ratio))
# Need to use 100 for embedding dimensions to match GloVe's vectors.
embedding_dim = 100

nb_words = len(vocab_to_int)
# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim))
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in GloVe, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding
        
# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))

# Note: The embeddings will be updated as the model trains, so our new 'random' embeddings will be more accurate by the end of training. 
#This is also why we want to only use words that appear at least 10 times. By having the model see the word numerous timesm it will be better able to understand what it means.
# Change the text from words to integers
# If word is not in vocab, replace it with <UNK> (unknown)
word_count = 0
unk_count = 0
int_headlines = []

for data in clean_headlines:
    int_daily_headlines = []
    for headline in data:
        int_headline = []
        for word in headline.split():
            word_count += 1
            if word in vocab_to_int:
                int_headline.append(vocab_to_int[word])
            else:
                int_headline.append(vocab_to_int["<UNK>"])
                unk_count += 1
        int_daily_headlines.append(int_headline)
    int_headlines.append(int_daily_headlines)

unk_percent = round(unk_count/word_count,4)*100 
print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))

# Find the length of headlines
lengths = []
for data in int_headlines:
    for headline in data:
        lengths.append(len(headline))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])
lengths.describe()

# Limit the length of a day's news to 200 words, and the length of any headline to 16 words.
# These values are chosen to not have an excessively long training time and 
# balance the number of headlines used and the number of words from each headline.
max_headline_length = 16
max_daily_length = 200
pad_headlines = []

for data in int_headlines:
    pad_daily_headlines = []
    for headline in data:
        # Add headline if it is less than max length
        if len(headline) <= max_headline_length:
            for word in headline:
                pad_daily_headlines.append(word)
        # Limit headline if it is more than max length  
        else:
            headline = headline[:max_headline_length]
            for word in headline:
                pad_daily_headlines.append(word)

# Pad daily_headlines if they are less than max length
    if len(pad_daily_headlines) < max_daily_length:
        for i in range(max_daily_length-len(pad_daily_headlines)):
            pad = vocab_to_int["<PAD>"]
            pad_daily_headlines.append(pad)
    # Limit daily_headlines if they are more than max length
    else:
        pad_daily_headlines = pad_daily_headlines[:max_daily_length]
    pad_headlines.append(pad_daily_headlines)

# Normalize opening prices (target values)
max_price = max(price)
min_price = min(price)
mean_price = np.mean(price)
def normalize(price):
    return ((price-min_price)/(max_price-min_price))

norm_price = []
for p in price:
    norm_price.append(normalize(p))

# Check that normalization worked well
print(min(norm_price))
print(max(norm_price))
print(np.mean(norm_price))

# Split data into training and testing sets.
# Validating data will be created during training.
x_train, x_test, y_train, y_test = train_test_split(pad_headlines, norm_price, 
                                                    test_size = 0.15, random_state = 2)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Check the lengths
print(len(x_train))
print(len(x_test))

filter_length1 = 3
filter_length2 = 5
dropout = 0.5
learning_rate = 0.001
weights = initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2)
nb_filter = 16
rnn_output_size = 128
hidden_dims = 128
wider = True
deeper = True

if wider == True:
    nb_filter *= 2
    rnn_output_size *= 2
    hidden_dims *= 2
    
def build_model():
    
    model1 = Sequential()
    
    model1.add(Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length))
    model1.add(Dropout(dropout))
    
    model1.add(Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length1, 
                             padding = 'same',
                            activation = 'relu'))
    model1.add(Dropout(dropout))
    
    if deeper == True:
        model1.add(Convolution1D(filters = nb_filter, 
                                 kernel_size = filter_length1, 
                                 padding = 'same',
                                activation = 'relu'))
        model1.add(Dropout(dropout))
    
    model1.add(LSTM(rnn_output_size, 
                   activation=None,
                   kernel_initializer=weights,
                   dropout = dropout))
    
    ####

    model2 = Sequential()
    
    model2.add(Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length))
    model2.add(Dropout(dropout))
    
    
    model2.add(Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length2, 
                             padding = 'same',
                             activation = 'relu'))
    model2.add(Dropout(dropout))
    
    if deeper == True:
        model2.add(Convolution1D(filters = nb_filter, 
                                 kernel_size = filter_length2, 
                                 padding = 'same',
                                 activation = 'relu'))
        model2.add(Dropout(dropout))
    
    model2.add(LSTM(rnn_output_size, 
                    activation=None,
                    kernel_initializer=weights,
                    dropout = dropout))
    
    ####
    merged_layers = concatenate([model1.output, model2.output])
    x = Dense(hidden_dims, kernel_initializer=weights)(merged_layers)
    x = Dropout(dropout)(x)
    '''if deeper == True:
        x = Dense(hidden_dims//2, kernel_initializer=weights)(x)
        x = Dropout(dropout)(x)
    else:'''
    x = Dense(1, kernel_initializer=weights, name='output')(x)
    x = Dropout(dropout)(x)

    #model = Sequential()

    model = Model([model1.input, model2.input], [x])

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr = learning_rate, clipvalue = 1.0), metrics=['accuracy'])
    return model

# Use grid search to help find a better model
for deeper in [False]:
    for wider in [True, False]:
        for learning_rate in [0.001]:
            for dropout in [0.3, 0.5]:
                model = build_model()
                print()
                print("Current model: Deeper={}, Wider={}, LR={}, Dropout={}".format(
                    deeper,wider,learning_rate,dropout))
                print()
                save_best_weights = 'XXXXX/question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}.h5'.format(
                    deeper,wider,learning_rate,dropout)

                callbacks = [ModelCheckpoint(save_best_weights, verbose=1, monitor='val_loss', save_best_only=True),
                             EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto'),
                             ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3)]

                history = model.fit([x_train,x_train],
                                    y_train,
                                    batch_size=128,
                                    epochs=5,
                                    validation_split=0.15,
                                    verbose=True,
                                    shuffle=True,
                                    callbacks = callbacks)

##Save model
#model.save_weights("XXXXXX/save_best_weights.h5")                

# Make predictions with the best weights
deeper=False
wider=True
dropout=0.3
learning_Rate = 0.001
# Need to rebuild model in case it is different from the model that was trained most recently.
model = build_model()
model.load_weights('XXXXX/question_pairs_weights_deeper=False_wider=True_lr=0.001_dropout=0.3.h5')
predictions = model.predict([x_test,x_test], verbose = True)

# Compare testing loss to training and validating loss
mse(y_test, predictions)

def unnormalize(price):
    '''Revert values to their unnormalized amounts'''
    price = price*(max_price-min_price)+min_price
    return(price)

unnorm_predictions = []
for pred in predictions:
    unnorm_predictions.append(unnormalize(pred))
    
unnorm_y_test = []
for y in y_test:
    unnorm_y_test.append(unnormalize(y))
    
# Calculate the median absolute error for the predictions
mae(unnorm_y_test, unnorm_predictions)

print("Summary of actual opening price changes")
print(pd.DataFrame(unnorm_y_test, columns=[""]).describe())
print()
print("Summary of predicted opening price changes")
print(pd.DataFrame(unnorm_predictions, columns=[""]).describe())

# Plot the predicted (blue) and actual (green) values
plt.figure(figsize=(15,5))
plt.plot(unnorm_predictions)
plt.plot(unnorm_y_test)
plt.title("Predicted (Blue) vs Actual (Amber) Opening Price Changes")
plt.xlabel("Testing instances")
plt.ylabel("Change in Opening Price")
plt.show()
