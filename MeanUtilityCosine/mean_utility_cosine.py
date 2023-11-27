import pandas as pd
import numpy as np
import sys

from Utilities.preprocessing import preprocess
from WeightedNNNCosine.weighted_nnn_cosine import n_nearest_neighbors

def main():
    """
    takes in filename, n value
    """
    # call preprocessing script, returns sparse dataframe
    argc = len(sys.argv)
    
    if argc < 2 or argc > 2:
        print("Usage: python mean_utility_cosine.py <Filename>")
    
    try:
        data, item_stats, user_stats = preprocess(sys.argv[1])
        
        if argc > 2:
            n = int(sys.argv[2])
        else:
            n = None
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(mean_utility(data, item_stats, user_stats, 10, 10))

def mean_utility(data, item_stats, user_stats, user_id, item_id):
    """
    given a user and item, use mean utility to predict the utility u(c,s) using the ratings matrix (data)
    mean utility = average utility across all users for the given item
    """
    return item_stats.loc[item_id, "mean_rating"]


if __name__ == "__main__":
    main()