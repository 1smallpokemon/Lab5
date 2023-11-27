import pandas as pd
import numpy as np
import sys

#from ..Utilities import preprocessing
from Utilities.preprocessing import preprocess

def main():
    """
    takes in filename, n value
    """
    # call preprocessing script, returns sparse dataframe
    argc = len(sys.argv)
    
    if argc < 3 or argc > 3:
        print("Usage: python hclustering.py <Filename> <n>")
    
    try:
        data, item_stats, user_stats = preprocess(sys.argv[1])
        
        if argc > 2:
            n = int(sys.argv[2])
        else:
            n = None
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    weighted_sum(data, item_stats, user_stats, 10, 10, 5)

def weighted_sum(data, item_stats, user_stats, user_id, item_id, n):
    """
    given a user and item, use weighted sum to predict the utility u(c,s) using the ratings matrix (data)
    """
    # use cosine similarity to determine n nearest neighbors
    neighbors = n_nearest_neighbors(data, user_id, n)
    user = data.iloc[user_id]
    #second_sigma = neighbors.apply(lambda x: cosine_similarity(user, x) * data.iloc[x.index, item_id], axis=1).sum()
    second_sigma = neighbors.apply(lambda x: cosine_similarity(user, x) * data.iloc[int(x.name), item_id], axis=1).sum()
    k = 1 / neighbors.apply(lambda x: abs(cosine_similarity(user, x)), axis=1).sum()
    return k * second_sigma

def n_nearest_neighbors(data, user_id, n):
    """
    given a user and ratings matrix (data), return the n nearest neighbors to the user
    return the neighbors as a sparse dataframe of just the rows corresponding to the n nearest neigbors
    """
    user = data.iloc[user_id]
    # sim_dict = dict() # dictionary mapping user id to similarity
    # for index,row in data.iterrows():
    #     sim = cosine_similarity(user, row)
    #     sim_dict[index] = sim

    similarities = data.apply(cosine_similarity, axis=1, c2=user)
    min_indices = similarities.argsort()[:n]

    return data.iloc[min_indices]
    

def cosine_similarity(c1, c2):
    """
    given the sparse vectors c1 and c2, compute and return the cosine similarity between them
    vectors are passed in as sparse series
    """
    # one row per user
    # for each column, multiply c1 value * c2 value, sum up those products
    # multiply the series element-wise and take the sum
    non_zero_indices = c1.index.intersection(c2.index)
    numerator = (c1[non_zero_indices] * c2[non_zero_indices]).sum()
    # square every value in c1 and add them, square every value in c2 and add them, multiply those two, sqrt everything
    denominator = np.sqrt((c1[non_zero_indices] ** 2).sum() * (c2[non_zero_indices] ** 2).sum())
    # divide the 1st result / 2nd result (above)
    return numerator / denominator 

if __name__ == "__main__":
    main()