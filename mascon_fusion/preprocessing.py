"""
Module: preprocessing.py

Description:
Utilities for preparing mascon time series prior to fusion and analysis.
Includes helpers to build a monthly decimal-year axis, remove a reference
epoch mean, and interpolate series onto a target time axis.

Functions:
- build_decyear_array
- remove_mean_epoch
- interpolate_timeseries

Author: R McGirr 2026-03
"""

import numpy as np
from scipy.interpolate import interp1d
from .time_utils import YMDtoYearFraction

def build_decyear_array(start_year=2002, end_year=2025, day=15):
    """
    Build a monthly decyear array using mid-month dates.
    
    Parameters:
    -----------
    start_year : int
        First year (inclusive)
    end_year : int
        Last year (inclusive)
    day : int
        Day of month to use (default: 15)
    """
    decyears = []
    for yr in range(start_year, end_year + 1):
        for month in range(1, 13):
            decyears.append(YMDtoYearFraction(yr, month, day))
    return np.array(decyears)

def remove_mean_epoch(data, decyear, start_year=2004, end_year=2010):
    """
    Remove mean value over a specified epoch period from time series data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Time series data, can be 1D (ntimes,) or 2D (ntimes, nmascons)
    decyear : numpy.ndarray
        Decimal year values corresponding to time dimension
    start_year : float
        Start year for mean calculation (default: 2004)
    end_year : float
        End year for mean calculation (default: 2010)
    
    Returns:
    --------
    data_demean : numpy.ndarray
        Data with mean removed
    mean_val : numpy.ndarray
        Mean value(s) that were removed
    """
    # Find indices for the mean epoch
    mean_idx = (decyear >= start_year) & (decyear < end_year)
    
    # Calculate mean over the specified epoch
    mean_val = np.mean(data[mean_idx], axis=0)
    
    # Remove mean
    data_demean = data - mean_val
    
    return data_demean, mean_val

def interpolate_timeseries(time, data, target_time, max_gap=None, kind="linear", fill_value=np.nan):
    """
    Interpolate time series to target_time. If max_gap is set, any target point
    farther than max_gap from the nearest observation is set to NaN.
    """
    time = np.asarray(time)
    target_time = np.asarray(target_time)
    data = np.asarray(data)

    f = interp1d(time, data, axis=0, kind=kind, bounds_error=False, fill_value='extrapolate')
    out = f(target_time)

    if max_gap is not None:
        nearest = np.min(np.abs(time[:, None] - target_time[None, :]), axis=0)
        mask = nearest > max_gap
        if out.ndim == 1:
            out[mask] = np.nan
        else:
            out[mask, :] = np.nan

    return out
