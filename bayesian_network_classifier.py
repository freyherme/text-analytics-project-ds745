

############################################
#             Import Libraries             #
############################################

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sqlite3 import Error
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import pickle
import numpy as np
from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 10)

############################################
#               Import Data                #
############################################
# Note:
# Detractors: Rating 1-6, Passive: 7-8, Promoters: 9:10

nps_data_2018 = pd.read_csv('Fall 2018 DnA NPS.csv')
nps_data_2021 = pd.read_csv('Fall 2021 DnA NPS.csv')

############################################
#               Prepare Data               #
############################################

# Drop unnecessary columns
nps_data_2018 = nps_data_2018.drop('Account ID', 1)
nps_data_2021 = nps_data_2021.drop('Channel', 1)

# Concat data frames
nps_data_full = pd.concat([nps_data_2018, nps_data_2021])

# Filter rows w/o response
nps_data_w_response = nps_data_full[nps_data_full['Response'].notna()]


# Encode Ratings:
#    0: Detractor (Rating 0-6)
#    1: Passive (Rating 7-8)
#    2: Promoter (Rating 9-10)

def encode_rating(rating):
    if rating > 8:
        return 2
    elif rating > 6:
        return 1
    elif rating >= 0:
        return 0
    else:
        return None


print("Encoding ratings...")
nps_data_w_response['nps_status'] = nps_data_w_response['Rating'].apply(encode_rating)
print(nps_data_w_response.head())

# Text Pre-Processing
tokenizer = nltk.WordPunctTokenizer()
stemmer = SnowballStemmer(language = 'english')
stop_words = nltk.corpus.stopwords.words('english')
custom_stop_words = ['illuminate', 'program', 'system', 'platform', 'would',
                     'use', 'used', 'also', 'thing', 'im', 'get', 'still']
stop_words = stop_words + custom_stop_words


def prepare_text(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()

    # Transform User Friendly / Not User Friendly Text
    doc = doc.replace("not user friendly", "not_user_friendly")
    doc = doc.replace("user friendly", "user_friendly")
    doc = doc.replace("userfriendly", "user_friendly")

    # Tokenize
    tokens = tokenizer.tokenize(doc)

    # Remove Stop Words
    tokens = [w for w in tokens if w not in stop_words]

    # Stem

    tokens = [stemmer.stem(w) for w in tokens]

    # Compile Document
    doc = ' '.join(tokens)
    return doc


prepare_dataset = np.vectorize(prepare_text)

# Process NPS Responses
print("Process NPS Responses...")
preprocessed_responses = prepare_dataset(nps_data_w_response['Response'])
nps_data_w_response['preprocessed_response'] = preprocessed_responses
# nps_data_w_response.to_csv('nps_data_w_preprocessed_response.csv')

# Create Bag-of-Words/Phrases Matrix
print("Create Bag-of-Words/Phrases Matrix...")
cv = CountVectorizer(ngram_range = (1, 2), min_df = 1, max_df = 1.0)
word_matrix = cv.fit_transform(nps_data_w_response['preprocessed_response'])
word_matrix = word_matrix.toarray()
words = cv.get_feature_names()
word_matrix_df = pd.DataFrame(word_matrix, columns = words)
print(word_matrix_df.sum())

# training_data_df = nps_data_w_response[['preprocessed_response', 'Rating']]
# training_data = training_data_df.apply(tuple, axis = 1)

############################################
#               Explore Data               #
############################################
# Distribution
print("Distribution... ")
nps_data_w_response.groupby('Rating').Response.count().plot.bar(ylim = 0)
plt.show()
nps_data_w_response.groupby('nps_status').Response.count().plot.bar(ylim = 0)
plt.show()

############################################
#               Train Model                #
############################################

print("Train/Test Split...")
trainX, testX, trainY, testY = train_test_split(word_matrix, nps_data_w_response['nps_status'])

print("Train classifier...")
mnb = MultinomialNB()
classifier = mnb.fit(X = trainX, y = trainY)

############################################
#              Evaluate Model              #
############################################

print("Predict...")
y_pred = mnb.predict(testX)

print("Confusion matrix...")
cm = confusion_matrix(y_true = testY, y_pred = y_pred)
print(cm)

print(classification_report(testY, y_pred))

############################################
#           Interpret Results              #
############################################

coefs_with_fns = sorted(zip(classifier.coef_[0], words))
top = zip(coefs_with_fns[:20], coefs_with_fns[:-(20 + 1):-1])
for (coef_1, fn_1), (coef_2, fn_2) in top:
    print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

class0_prob_sorted = classifier.feature_log_prob_[0, :].argsort()[::-1]
class1_prob_sorted = classifier.feature_log_prob_[1, :].argsort()[::-1]
class2_prob_sorted = classifier.feature_log_prob_[2, :].argsort()[::-1]
# class3_prob_sorted = classifier.feature_log_prob_[3, :].argsort()[::-1]
# class4_prob_sorted = classifier.feature_log_prob_[4, :].argsort()[::-1]
# class5_prob_sorted = classifier.feature_log_prob_[5, :].argsort()[::-1]
# class6_prob_sorted = classifier.feature_log_prob_[6, :].argsort()[::-1]
# class7_prob_sorted = classifier.feature_log_prob_[7, :].argsort()[::-1]
# class8_prob_sorted = classifier.feature_log_prob_[8, :].argsort()[::-1]
# class9_prob_sorted = classifier.feature_log_prob_[9, :].argsort()[::-1]
# class10_prob_sorted = classifier.feature_log_prob_[9, :].argsort()[::-1]

print("class0... ")
print(np.take(words, class0_prob_sorted[:20]))

print("class1... ")
print(np.take(words, class1_prob_sorted[:20]))

print("class2... ")
print(np.take(words, class2_prob_sorted[:20]))

# print("class3... ")
# print(np.take(words, class3_prob_sorted[:10]))
#
# print("class4... ")
# print(np.take(words, class4_prob_sorted[:10]))
#
# print("class5... ")
# print(np.take(words, class5_prob_sorted[:10]))
#
# print("class6... ")
# print(np.take(words, class6_prob_sorted[:10]))
#
# print("class7... ")
# print(np.take(words, class7_prob_sorted[:10]))
#
# print("class8... ")
# print(np.take(words, class8_prob_sorted[:10]))
#
# print("class9... ")
# print(np.take(words, class9_prob_sorted[:10]))
#
# print("class10... ")
# print(np.take(words, class10_prob_sorted[:10]))

print("fin.")
