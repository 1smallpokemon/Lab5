import pandas as pd
import numpy as np
import sys
import random
from Utilities.preprocessing import preprocess
from AvgNNNPearson.avg_nnn_pearson import *
from MeanUtility.mean_utility import *
from WeightedNNNCosine.weighted_nnn_cosine import *
from WeightedSumPearson.weighted_sum_pearson import *
from Utilities.mean_absolute_error import mean_absolute_error

def evaluate_cf_random(method, size, repeats, data, item_stats, user_stats):
    # Randomly sample test cases and evaluate the specified method
    # ...
    user_count = len(data)
    item_count = len(data.columns)

    for i in range(repeats):
        predictions = []
        for j in range(size):
            user_id, item_id = generate_pair(data, user_count, item_count)

            replace = False
            if not data.iloc[user_id, item_id] is np.nan:
                old_value = data.iloc[user_id, item_id]
                sparse_type = data.dtypes[item_id]
                data[item_id] = data[item_id].sparse.to_dense()
                data.loc[user_id, item_id] = np.nan
                data[item_id] = data[item_id].astype(sparse_type)
                replace = True
            
            if method == 'mean_utility':
                prediction = mean_utility(data, item_stats, user_stats, user_id, item_id)
            elif method == 'nnn_weighted_sum_cosine':
                prediction = weighted_sum(user_id, item_id, data, item_stats, user_stats)
            elif method == 'nnn_avg_pearson':
                prediction = predict_rating_n_nearest_neighbors(user_id, item_id, data, n_neighbors = 10)
            elif method == 'nnn_weighted_sum_pearson':
                apple = 'orange' 
            else:
                raise ValueError("Unknown method")
            
            if replace:
                sparse_type = data.dtypes[item_id]
                data[item_id] = data[item_id].sparse.to_dense()
                data.loc[user_id, item_id] = old_value
                data[item_id] = data[item_id].astype(sparse_type)
            
            actual_rating = data.at[user_id, item_id]
            predictions.append((user_id, item_id, actual_rating, prediction))

        # Compute MAE and other statistics here
        # output format: userID, itemID, Actual_Rating, Predicted_Rating, Delta_Rating
        print("userID, itemID, Actual_Rating, Predicted_Rating, Delta_Rating")
        for row in predictions:
            print(str(row[0]), "," + str(row[1]), "," + str(row[2]), ",", str(row[3]), ",", str(row[2] - row[3]))
        print(mean_absolute_error(predictions))

def generate_pair(data, user_count, item_count):
    """
    return (userId, itemId)
    """
    c = random.randint(0, user_count - 1)
    s = random.randint(0, item_count - 1)
    if data.iloc[c,s] is np.nan:
        return generate_pair(data, user_count, item_count)
    return (c, s)

if __name__ == "__main__":
    # Basic command line interface
    methods = ['mean_utility',
                'nnn_weighted_sum_cosine',
                'nnn_avg_pearson',
                'nnn_weighted_sum_pearson']
    
    if len(sys.argv) != 4:
        print("Usage: python EvaluateCFList.py <Method> <Size> <Repeats>")
        print(methods)
        sys.exit(1)
    
    method = methods[int(sys.argv[1])]
    size = int(sys.argv[2])
    repeats = int(sys.argv[3])
    
    # Preprocess data
    data, item_stats, user_stats = preprocess('data/jester-data-1.csv')
    
    # Run evaluation
    evaluate_cf_random(method, size, repeats, data, item_stats, user_stats)