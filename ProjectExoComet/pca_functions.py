import pandas as pd  
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_funct(df, n_componets): 
    """preform PCA on the data"""
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Apply PCA
    pca = PCA(n_components=n_componets)  # Number of components to keep
    principal_components = pca.fit_transform(scaled_data)
    return pca, principal_components, scaler, scaled_data

def pca_estimate(pca, principal_components):
    # """ estimate the original data """"
    estimated_original_data = pca.inverse_transform(principal_components)

    # Convert the estimated data back to a DataFrame
    estimated_df = pd.DataFrame(estimated_original_data)

    return estimated_df

def vaiations(scaled_data, estimated_df, df, snr):
    """ find the variations in the data and return the scaled data with variations larger than the snr"""
    difference_df = pd.DataFrame(scaled_data - estimated_df)
    index_list = []
    for index, row in difference_df.iterrows():
        if max(abs(row)) > snr:
            index_list.append(index)
    return index_list
