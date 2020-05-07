import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

def sample_recommendation(model, train_data, labels, user_ids, n_known, n_results):
    n_users, n_items = train_data.shape
    for user_id in user_ids:
        known_positives = labels[train_data.tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = labels[np.argsort(-scores)]
        top_items_filtered = [item for item in top_items if item not in known_positives]
        print("User %s" % user_id)
        print("    Known positives:")
        for x in known_positives[:n_known]:
            print("        %s" % x)
        print("    Recommended:")
        for x in top_items_filtered[:n_results]:
            print("        %s" % x)

# fetch items
data = fetch_movielens(min_rating = 4.0)
# take train data and test data
train_set = data['train']
test_set = data['test']
# take labels for all items
labels = data['item_labels']
# show data
print(repr(train_set))
print(repr(test_set))

# define model
model = LightFM(loss = 'warp')
# train model
model.fit(train_set, epochs=30, num_threads=2)

# get recommendations
users_ids_list = [3, 25, 451, 737, 901]
known_items_to_show = 5
recommendations_to_show = 3
sample_recommendation(model, train_set, labels, users_ids_list, known_items_to_show, recommendations_to_show)
