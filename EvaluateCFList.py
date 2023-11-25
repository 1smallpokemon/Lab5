import pandas as pd
import numpy as np
import sys
from Utilities.preprocessing import preprocess
from AvgNNNPearson.avg_nnn_pearson import  *
from MeanUtilityCosine.mean_utility_cosine import *
from WeightedNNNCosine.weighted_nnn_cosine import *
from WeightedSumPearson.weighted_sum_pearson import *

def evaluate_cf_list(method, test_cases_filename, data, item_stats, user_stats):
    # Load test cases
    test_cases = pd.read_csv(test_cases_filename)
    predictions = []
    
    for _, row in test_cases.iterrows():
        user_id, item_id = row
        if method == 'mean_utility':
            prediction = mean_utility(user_id, item_id, data, item_stats)
        elif method == 'weighted_sum':
            prediction = weighted_sum(user_id, item_id, data, item_stats, user_stats)
        # Add other methods here
        else:
            raise ValueError("Unknown method")
        
        actual_rating = data.at[user_id, item_id]
        predictions.append((user_id, item_id, actual_rating, prediction))
    
    # Compute MAE and other statistics here
    # ...

if __name__ == "__main__":
    # Basic command line interface
    if len(sys.argv) != 3:
        print("Usage: python EvaluateCFList.py <Method> <TestCasesFilename>")
        sys.exit(1)
    
    method = sys.argv[1]
    test_cases_filename = sys.argv[2]
    
    # Preprocess data
    data, item_stats, user_stats = preprocess('path_to_jester_data.csv')
    
    # Run evaluation
    evaluate_cf_list(method, test_cases_filename, data, item_stats, user_stats)
