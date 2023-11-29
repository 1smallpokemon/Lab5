import pandas as pd
import numpy as np

def mean_absolute_error(predictions):
    """
    Calculate the Mean Absolute Error (MAE) given predictions: 
    a list of tuples (user_id, item_id, actual_rating, prediction).
    """
    # Filter out the predictions that are NaN
    valid_predictions = [p for p in predictions if not np.isnan(p[3])]
    
    # Calculate the sum of absolute errors for non-NaN predictions
    error_sum = sum(abs(p[2] - p[3]) for p in valid_predictions)
    
    # Calculate the mean absolute error
    mae = error_sum / len(valid_predictions) if valid_predictions else np.nan
    return mae