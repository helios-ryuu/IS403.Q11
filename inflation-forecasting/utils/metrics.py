"""
Evaluation Metrics for Inflation Forecasting Models
Author: IS403.Q11 Project
Date: November 2025

This module provides evaluation metrics:
- RMSFE (Root Mean Squared Forecast Error)
- MAPE (Mean Absolute Percentage Error)
"""

import numpy as np


def rmsfe(y_true, y_pred):
    """
    Calculate Root Mean Squared Forecast Error (RMSFE)
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float
        RMSFE value
    
    Formula:
    --------
    RMSFE = sqrt(mean((y_true - y_pred)^2))
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float
        MAPE value (in percentage)
    
    Formula:
    --------
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Avoid division by zero
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance using both RMSFE and MAPE
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str, optional
        Name of the model (for display purposes)
    
    Returns:
    --------
    dict
        Dictionary containing RMSFE and MAPE values
    """
    rmsfe_value = rmsfe(y_true, y_pred)
    mape_value = mape(y_true, y_pred)
    
    results = {
        'model': model_name,
        'RMSFE': rmsfe_value,
        'MAPE': mape_value
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSFE: {rmsfe_value:.4f}")
    print(f"  MAPE:  {mape_value:.2f}%")
    
    return results
