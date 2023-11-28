import pandas as pd
import numpy as np
import sys
from Utilities.preprocessing import preprocess
from AvgNNNPearson.avg_nnn_pearson import *
from MeanUtility.mean_utility import *
from WeightedNNNCosine.weighted_nnn_cosine import *
from WeightedSumPearson.weighted_sum_pearson import *

def evaluate_cf_random(method, size, repeats, data, item_stats, user_stats):
    # Randomly sample test cases and evaluate the specified method
    # ...

if __name__ == "__main__":
    # Basic command line interface
    if len(sys.argv) != 4:
        print("Usage: python EvaluateCFRandom.py <Method> <Size> <Repeats>")
        sys.exit(1)
    
    method = sys.argv[1]
    size = int(sys.argv[2])
    repeats = int(sys.argv[3])
    
    # Preprocess data
    data, item_stats, user_stats = preprocess('path_to_jester_data.csv')
    
    # Run evaluation
    evaluate_cf_random(method, size, repeats, data, item_stats, user_stats)
