import pandas as pd
import numpy as np
import sys
import random
from Utilities.preprocessing import preprocess
from AvgNNNPearson.avg_nnn_pearson import *
from MeanUtility.mean_utility import *
from WeightedNNNCosine.weighted_nnn_cosine import *
from WeightedSumPearson.weighted_nnn_pearson import *
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
                prediction = weighted_sum(data, item_stats, user_stats, user_id, item_id, n=10)
            elif method == 'nnn_avg_pearson':
                prediction = nnn_avg_pearson(user_id, item_id, data, n_neighbors = 10)
            elif method == 'nnn_weighted_sum_pearson':
                prediction = nnn_weighted_sum_pearson(user_id, item_id, data, n_neighbors = 10)
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
        tp, fp, tn, fn = calculate_confusion_matrix(predictions)
        precision, recall, f1 = calculate_precision_recall_f1(tp, fp, tn, fn)
        
        print("userID,itemID,Actual_Rating,Predicted_Rating,Delta_Rating")
        for row in predictions:
            print(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[2] - row[3]}")
        
        mae = mean_absolute_error(predictions)
        print(f"MAE: {mae}")
        print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

# Function to calculate the confusion matrix and related statistics
def calculate_confusion_matrix(predictions):
    tp = fp = tn = fn = 0
    for _, _, actual, predicted in predictions:
        if actual >= 5 and predicted >= 5:
            tp += 1
        elif actual < 5 and predicted >= 5:
            fp += 1
        elif actual >= 5 and predicted < 5:
            fn += 1
        elif actual < 5 and predicted < 5:
            tn += 1
    return tp, fp, tn, fn

def calculate_precision_recall_f1(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

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