import pandas as pd
import numpy as np

def mean_absolute_error(predictions):
    """
    given predictions = array of tuples (user_id, item_id, actual_rating, prediction)
    """
    w = len(predictions)
    sigma = 0
    for pairing in predictions:
        prediction = pairing[3]
        actual_rating = pairing[2]
        sigma += abs(prediction - actual_rating)
    return sigma / w