import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import pyLDAvis
import pyLDAvis.sklearn


pd.options.display.max_colwidth = 200

# Import data
nps_data_2018 = pd.read_csv('Fall 2018 DnA NPS.csv')
nps_data_2021 = pd.read_csv('Fall 2021 DnA NPS.csv')

# Drop unnecessary columns
nps_data_2018 = nps_data_2018.drop('Account ID', 1)
nps_data_2021 = nps_data_2021.drop('Channel', 1)

# Concat data frames
nps_data_full = pd.concat([nps_data_2018, nps_data_2021])

# Filter rows w/o response
nps_data_w_response = nps_data_full[nps_data_full['Response'].notna()]

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

# Prepare NPS Responses
print("Prepare NPS Responses...")
preprocessed_responses = prepare_dataset(nps_data_w_response['Response'])
nps_data_w_response['preprocessed_response'] = preprocessed_responses
nps_data_w_response.to_csv('nps_data_w_preprocessed_response.csv')

# Create Bag-of-Words/Phrases Matrix
print("Create Bag-of-Words/Phrases Matrix...")
cv = CountVectorizer(ngram_range = (1, 3), min_df = 5, max_df = 1.0)
word_matrix = cv.fit_transform(nps_data_w_response['preprocessed_response'])
word_matrix_arr = word_matrix.toarray()
words = cv.get_feature_names()
word_matrix_df = pd.DataFrame(word_matrix_arr, columns = words)
print(word_matrix_df.sum())

# Create TF-IDF Matrix
# This is a normed and scaled matrix
# Ensures that high-frequency tokens don't overwhelm other tokens in the dataset

print("Create TF-IDF Matrix...")
tfid = TfidfVectorizer(ngram_range = (1, 2), min_df = 5, max_df = 0.7)
tfid_matrix = tfid.fit_transform(nps_data_w_response['preprocessed_response'])
tfid_matrix = tfid_matrix.toarray()
words_tfid = tfid.get_feature_names()
tfid_matrix_df = pd.DataFrame(tfid_matrix, columns = words_tfid)
print(tfid_matrix_df.max())


# Calculate Similarity Scores
print("Calculate Similarity Scores...")
sim_matrix = cosine_similarity(tfid_matrix)
sim_df = pd.DataFrame(sim_matrix)
print(sim_df.head())

# Clustering (Unsupervised)
# print("Clustering...")
# ward_linkage = linkage(sim_matrix, 'ward', optimal_ordering = True)
# cluster_df = pd.DataFrame(ward_linkage,
#                           columns = ['doc1_cluster1', 'doc2_cluster2', 'distance', 'cluster_size'],
#                           dtype = 'object')
# print(cluster_df.head(20))

# Create Dendogram
# print("Create Dendogram...")
# plt.figure(figsize=(8, 3))
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Data point')
# plt.ylabel('Distance')
# dendrogram(ward_linkage, p = 5, truncate_mode = 'level')
# plt.axhline(y=30.0, c='k', ls='--', lw=0.5)
# plt.show()

# Topic Model
print("Topic Model...")
lda = LatentDirichletAllocation(n_components = 5, max_iter = 10, random_state=0, n_jobs = 20, verbose = 1)
dt_matrix = lda.fit(word_matrix)
# features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3', 'T4', 'T5'])

# print("Analyzing topics...")
# tt_matrix = lda.components_
# for i, topic_weights in enumerate(tt_matrix):
#     topic = [(token, weight) for token, weight in zip(words, topic_weights)]
#     topic = sorted(topic, key=lambda x: -x[1])
#     topic = [item for item in topic if item[1] > 40]
#     print(f"Topic {i+1}:")
#     print(topic)
#     print()

prepared_ldavis = pyLDAvis.sklearn.prepare(dt_matrix, word_matrix, cv)
pyLDAvis.display(prepared_ldavis)















