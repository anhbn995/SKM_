import numpy as np

def get_nor_val(img):
    
    data = np.empty(img.shape)
    for i in range(img.shape[0]):
        
        img1 = np.where(img[i]>5000, np.nan, img[i]) 
        X_mean = np.nanmean(img1, axis=0)
        X_std = np.nanstd(img1, axis=0)

        X_std[X_std == 0] = 1  # Handle division by zero
        X_std[X_std < 1e-12] = 1e-12  # Handle division by small numbers
        X_standardized = (img[i] - X_mean) / X_std + 1

        # Scale the data to [0, 1]
        X_min = np.min(X_standardized, axis=0)
        X_max = np.max(X_standardized, axis=0)
        X_normalized = (X_standardized - X_min) / (X_max - X_min)
        data[i] = X_normalized 
    return data