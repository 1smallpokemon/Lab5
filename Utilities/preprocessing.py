import pandas as pd
import numpy as np

def load_data(file_path):
    # Load the data, replacing '99' with NaN to represent missing values
    data = pd.read_csv(file_path, header=None)
    data.replace(99, np.nan, inplace=True)

    # Convert to sparse type using SparseDtype
    for column in data.columns:
        data[column] = data[column].astype(pd.SparseDtype(float, np.nan))

    return data

def compute_item_stats(data):
    # Calculate the mean rating for each joke, skipping NaN values
    item_means = data.mean(axis=0, skipna=True)
    # Count the number of ratings for each joke, skipping NaN values
    item_counts = data.count(axis=0)
    
    # Create a DataFrame with the computed statistics
    item_stats = pd.DataFrame({'mean_rating': item_means, 'rating_count': item_counts})
    return item_stats

def compute_user_stats(data):
    # Calculate the mean rating for each user, skipping NaN values
    user_means = data.mean(axis=1, skipna=True)
    # Count the number of ratings for each user, skipping NaN values
    user_counts = data.count(axis=1)
    
    # Create a DataFrame with the computed statistics
    user_stats = pd.DataFrame({'mean_rating': user_means, 'rating_count': user_counts})
    return user_stats

def preprocess(file_path):
    # Load and preprocess the dataset
    data = load_data(file_path)
    
    # Compute statistics for items and users
    item_stats = compute_item_stats(data)
    user_stats = compute_user_stats(data)
    
    return data, item_stats, user_stats