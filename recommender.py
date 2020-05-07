import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

def sample_recommendation(model, data, user_ids, n_results):
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]
        print("User %s" % user_id)
        print("    Known positives:")
        for x in known_positives[:n_results]:
            print("        %s" % x)
        print("    Recommended:")
        for x in top_items[:2*n_results]:
            if x not in known_positives:
                print("        %s" % x)

# fetch items
data = fetch_movielens(min_rating = 4.0)
# take train data and test data
train_set = data['train']
test_set = data['test']
# show data
print(repr(train_set))
print(repr(test_set))

# define model
model = LightFM(loss = 'warp')
# train model
model.fit(train_set, epochs=30, num_threads=2)

# get recommendations
sample_recommendation(model, data, [3, 25, 451], 3)
