import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k

def sample_recommendation(model, train_data, item_labels, user_ids, n_known=3, n_results=2):
    n_users, n_items = train_data.shape
    for user_id in user_ids:
        known_positives = item_labels[train_data.tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = item_labels[np.argsort(-scores)]
        top_items_filtered = [item for item in top_items if item not in known_positives]
        print("User %s" % user_id)
        print("    Known positives:")
        for x in known_positives[:n_known]:
            print("        %s" % x)
        print("    Recommended:")
        for x in top_items_filtered[:n_results]:
            print("        %s" % x)

# fetch items with positive ratings (account for positive feedback only)
data = fetch_movielens(min_rating = 4.0)
# take train data and test data
train_set = data['train']
test_set = data['test']
# take labels for all items
item_labels = data['item_labels']
# show data
print(repr(train_set))
print(repr(test_set))

# define model (hybrid method)
model = LightFM(loss = 'warp',
                random_state=2016,
                no_components=100,
                user_alpha=0.000005)
# train model
model.fit(train_set, epochs=30, num_threads=2)

# get recommendations
users_ids_list = [3, 25, 451, 737, 901]
known_items_to_show = 5
recommendations_to_show = 3
sample_recommendation(model=model, train_data=train_set, item_labels=item_labels, user_ids=users_ids_list, n_known=known_items_to_show, n_results=recommendations_to_show)

patk = precision_at_k(model, test_set, train_interactions=train_set, k=recommendations_to_show,
               user_features=None, item_features=data['item_features'], preserve_rows=True, num_threads=1, check_intersections=True)

ratk = recall_at_k(model, test_set, train_interactions=train_set, k=recommendations_to_show,
               user_features=None, item_features=data['item_features'], preserve_rows=True, num_threads=1, check_intersections=True)

print('\nPrecision at k (proportion of recommended items in the top-k set that are relevant)')
print(patk[users_ids_list])

print('\nRecall at k (proportion of relevant items found in the top-k recommendations)')
print(ratk[users_ids_list])
