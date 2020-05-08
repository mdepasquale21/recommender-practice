# import dependent libraries
import pandas as pd
import os
from scipy.sparse import csr_matrix
import numpy as np
import warnings

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
#from skopt import forest_minimize
import random

#ignore warnings
warnings.filterwarnings("ignore")

###############################################################################################################################
# function to sample recommendations to a given user
def sample_recommendation_user(model, interactions, user_id, user_dict, 
                               item_dict,threshold = 0,nrec_items = 5, show = True):
    
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x, np.arange(n_items), item_features=books_metadata_csr))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                 [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        print ("User: " + str(user_id))
        print("Known Likes:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter+=1

        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
###############################################################################################################################

# import books metadata
books_metadata = pd.read_json('./data-books/goodreads_books_poetry.json', lines=True)
# import user-book interactions
interactions = pd.read_json('./data-books/goodreads_interactions_poetry.json', lines=True)

############################################################################################################################### work on books metadata
print('\nBooks metadata:')
print('Fields:')
print(books_metadata.columns.values)
print('Samples from dataset:')
print(books_metadata.sample(2))
print('Shape of books metadata')
print(books_metadata.shape)

# Limit the books metadata to selected fields
print('Select only some fields from books metadata dataset')
books_metadata_selected = books_metadata[['book_id', 'average_rating', 'is_ebook', 'num_pages', 
                                          'publication_year', 'ratings_count', 'language_code']]
print(books_metadata_selected.sample(2).to_string(index=False))

print('Perform some manipulation on data...')
# using pandas cut method to convert fields into discrete intervals
books_metadata_selected['num_pages'].replace(np.nan, -1, inplace=True)
books_metadata_selected['num_pages'].replace('', -1, inplace=True)
books_metadata_selected['num_pages'] = pd.to_numeric(books_metadata_selected['num_pages'])
books_metadata_selected['num_pages'] = pd.cut(books_metadata_selected['num_pages'], bins=25)
# rounding ratings to neares .5 score
books_metadata_selected['average_rating'] = books_metadata_selected['average_rating'].apply(lambda x: round(x*2)/2)
# using pandas qcut method to convert fields into quantile-based discrete intervals
books_metadata_selected['ratings_count'] = pd.qcut(books_metadata_selected['ratings_count'], 25)
# replacing missing values to year 2100
books_metadata_selected['publication_year'].replace(np.nan, 2100, inplace=True)
books_metadata_selected['publication_year'].replace('', 2100, inplace=True)
# replacing missing values to 'unknown'
books_metadata_selected['language_code'].replace(np.nan, 'unknown', inplace=True)
books_metadata_selected['language_code'].replace('', 'unknown', inplace=True)
books_metadata_selected['language_code'].replace('en-GB', 'eng', inplace=True)
books_metadata_selected['language_code'].replace('en-US', 'eng', inplace=True)
books_metadata_selected['language_code'].replace('it-IT', 'ita', inplace=True)
# convert is_ebook column into 1/0 where true=1 and false=0
books_metadata_selected['is_ebook'] = books_metadata_selected.is_ebook.map(lambda x: 1*(x == 'true'))
print('Books metadata after some manipulation')
print(books_metadata_selected.sample(5).to_string(index=False))

############################################################################################################################### work on interactions data
print('\nInteractions data:')
print('Fields')
print(interactions.columns.values)
print('Some samples from dataset')
print(interactions.sample(2).to_string(index=False))
print('Shape of interactions data')
print(interactions.shape)

# Limit the interactions data to selected fields
print('Select only some fields from interactions dataset')
interactions_selected = interactions[['user_id', 'book_id', 'is_read', 'rating']]
print(interactions_selected.sample(2).to_string(index=False))

print('Perform some manipulation on data...')
# convert is_read column into 1/0 where True=1 and False=0
interactions_selected['is_read'] = interactions_selected.is_read.map(lambda x: 1*(x == True))
print('Interaction data after some manipulation')
print(interactions_selected.sample(5).to_string(index=False))

# Since we have two fields denoting interaction between a user and a book, is_read and rating
# let's see how many data points we have where the user hasn't read the book but have given the ratings.
print('\nUsers ratings (columns) divided by having actually read the book (2 rows)')
interactions_counts = interactions_selected.groupby(['rating', 'is_read']).size().reset_index().pivot(columns='rating', index='is_read', values=0)
print(interactions_counts.to_string())

# From the above results, we can conclusively infer that users with ratings >= 1 have all read the book.
# Therefore, we'll use the ratings as the final score, drop interactions where is_read is false,
# and limit interactions from random 500 users to limit the data size for further analysis
print('\nSelect only user-item interactions with users that have actually read the book, and a limited number (to simplify calculations)')
interactions_selected = interactions_selected.loc[interactions_selected['is_read']==1, ['user_id', 'book_id', 'rating']]
interactions_selected = interactions_selected[interactions_selected['user_id'].isin(random.sample(list(interactions_selected['user_id'].unique()), k=5000))]
print(interactions_selected.sample(10).to_string(index=False))
print('Final interactions dataset shape')
print(interactions_selected.shape)

############################################################################################################################### data processing

# Now, let's transform the available data into CSR sparse matrix that can be used for matrix operations.
# We will start by the process by creating books_metadata matrix which is np.float64 csr_matrix of shape ([n_books, n_books_features])
# Each row contains that book's weights over features. However, before we create a sparse matrix, we'll first create a item dictionary for future references

item_dict ={}
df = books_metadata[['book_id', 'title']].sort_values('book_id').reset_index()

for i in range(df.shape[0]):
    item_dict[(df.loc[i,'book_id'])] = df.loc[i,'title']
    
# dummify categorical features
books_metadata_selected_transformed = pd.get_dummies(books_metadata_selected, columns = ['average_rating', 'is_ebook', 'num_pages', 
                                                                                         'publication_year', 'ratings_count', 
                                                                                         'language_code'])

books_metadata_selected_transformed = books_metadata_selected_transformed.sort_values('book_id').reset_index().drop('index', axis=1)
print('First rows of books metadata transformed')
print(books_metadata_selected_transformed.head(5))

# convert to csr matrix
books_metadata_csr = csr_matrix(books_metadata_selected_transformed.drop('book_id', axis=1).values)
print('Sparse matrix of books metadata')
print(repr(books_metadata_csr))

# Next we'll create an iteractions matrix which is np.float64 csr_matrix of shape ([n_users, n_books]).
# We'll also create a user dictionary for future use cases
user_book_interaction = pd.pivot_table(interactions_selected, index='user_id', columns='book_id', values='rating')
# fill missing values with 0
user_book_interaction = user_book_interaction.fillna(0)
print('First rows of interactions data transformed')
print(user_book_interaction.head(5))

user_id = list(user_book_interaction.index)
user_dict = {}
counter = 0 
for i in user_id:
    user_dict[i] = counter
    counter += 1

# convert to csr matrix
user_book_interaction_csr = csr_matrix(user_book_interaction.values)
print('Sparse matrix of interactions data')
print(repr(user_book_interaction_csr))

############################################################################################################################### model

model = LightFM(loss='warp', random_state=2016, learning_rate=0.90, no_components=150, user_alpha=0.000005)

model = model.fit(user_book_interaction_csr, epochs=100, num_threads=16, verbose=False)

sample_recommendation_user(model, user_book_interaction, interactions_selected['user_id'].iloc[0], user_dict, item_dict)
