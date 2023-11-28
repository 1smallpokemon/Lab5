import pandas as pd
import numpy as np

def pearson_correlation(vector_x, vector_y):
    """
    Calculate the Pearson correlation coefficient between two vectors, x and y, repersenting users.
    """
    mean_x = np.nanmean(vector_x)
    mean_y = np.nanmean(vector_y)

    # Center the vectors by subtracting their means
    centered_x = vector_x - mean_x
    centered_y = vector_y - mean_y

    # Compute the covariance and the standard deviations
    covariance = np.nansum(centered_x * centered_y)
    std_x = np.sqrt(np.nansum(centered_x ** 2))
    std_y = np.sqrt(np.nansum(centered_y ** 2))

    # Check for zero standard deviation and return the correlation
    if std_x == 0 or std_y == 0:
        return 0  # Return 0 if no variation in one of the vectors
    else:
        correlation = covariance / (std_x * std_y)
        return correlation

def find_n_nearest_neighbors(user_id, data, n_neighbors):
    """
    Find the N nearest neighbors of a given user based on Pearson correlation.
    """
    # Compute Pearson correlation for the user with all others
    user_vector = data.loc[user_id]
    correlations = {}
    for other_user_id in data.index:
        if other_user_id != user_id:
            other_user_vector = data.loc[other_user_id]
            correlation = pearson_correlation(user_vector, other_user_vector)
            if not np.isnan(correlation):
                correlations[other_user_id] = correlation
        
    
    # Sort users by correlation and select the top N neighbors
    sorted_neighbors = sorted(correlations.items(), key=lambda item: item[1], reverse=True)
    top_neighbors = sorted_neighbors[:n_neighbors]
    return top_neighbors

def nnn_weighted_sum_pearson(user_id, item_id, data, n_neighbors):
    """
    Predict the rating using the N nearest neighbors.
    """
    neighbors = find_n_nearest_neighbors(user_id, data, n_neighbors)
    # Normalize the correlation coefficients
    sum_abs_correlations = sum(abs(neighbor[1]) for neighbor in neighbors)
    k = 1 / sum_abs_correlations if sum_abs_correlations != 0 else 0
    
    # Calculate the weighted sum of neighbors' ratings
    weighted_sum = sum(neighbor[1] * data.at[neighbor[0], item_id] for neighbor in neighbors)
    prediction = k * weighted_sum
    return prediction