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
print(books_metadata_selected.sample(2))

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
print(books_metadata_selected.sample(5))

############################################################################################################################### work on interactions data
print('\nInteractions data:')
print('Fields')
print(interactions.columns.values)
print('Some samples from dataset')
print(interactions.sample(2))
print('Shape of interactions data')
print(interactions.shape)

# Limit the interactions data to selected fields
print('Select only some fields from interactions dataset')
interactions_selected = interactions[['user_id', 'book_id', 'is_read', 'rating']]
print(interactions_selected.sample(2))

print('Perform some manipulation on data...')
# convert is_read column into 1/0 where True=1 and False=0
interactions_selected['is_read'] = interactions_selected.is_read.map(lambda x: 1*(x == True))
print('Interaction data after some manipulation')
print(interactions_selected.sample(5))

# Since we have two fields denoting interaction between a user and a book, is_read and rating
# let's see how many data points we have where the user hasn't read the book but have given the ratings.
print('\nUsers ratings (columns) divided by having actually read the book (2 rows)')
interactions_counts = interactions_selected.groupby(['rating', 'is_read']).size().reset_index().pivot(columns='rating', index='is_read', values=0)
print(interactions_counts.to_string())

# From the above results, we can conclusively infer that users with ratings >= 1 have all read the book.
# Therefore, we'll use the ratings as the final score, drop interactions where is_read is false,
# and limit interactions from random 500 users to limit the data size for further analysis
