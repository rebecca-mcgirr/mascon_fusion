"""Utilities for preparing mascon time series prior to fusion and analysis."""

import copy
import datetime as dt
import numpy as np
from scipy.interpolate import interp1d
from .time_utils import YMDtoYearFraction

REF_DATE = dt.datetime(2002, 1, 1)


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


def days_to_decyear(days, ref_date=REF_DATE):
    """Convert days since ref_date to decimal year, preserving fractional days."""
    values = np.asarray(days, dtype=float)
    out = np.empty(values.shape, dtype=float)

    for index, value in np.ndenumerate(values):
        date = ref_date + dt.timedelta(days=float(value))
        year_start = dt.datetime(date.year, 1, 1)
        next_year_start = dt.datetime(date.year + 1, 1, 1)
        out[index] = date.year + (date - year_start).total_seconds() / (
            next_year_start - year_start
        ).total_seconds()

    return float(out) if np.isscalar(days) else out


def valid_interpolation_targets(source_time, target_time, extrapolate_days=0.0):
    """Return True where target_time falls within the allowed source range."""
    source_time = np.asarray(source_time, dtype=float)
    target_time = np.asarray(target_time, dtype=float)

    return (
        (target_time >= np.nanmin(source_time) - extrapolate_days)
        & (target_time <= np.nanmax(source_time) + extrapolate_days)
    )


def interpolate_mascon_ewh(
    source_time,
    source_ewh,
    target_time,
    extrapolate_days=0.0,
    kind="linear",
    dtype=np.float32,
):
    """Interpolate mascon EWH time series to target_time along axis 0."""
    source_time = np.asarray(source_time, dtype=float)
    target_time = np.asarray(target_time, dtype=float)
    source_ewh = np.asarray(source_ewh, dtype=dtype)

    order = np.argsort(source_time)
    x = source_time[order]
    y = source_ewh[order]

    interpolator = interp1d(
        x,
        y,
        axis=0,
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    out = interpolator(target_time).astype(dtype)

    valid = valid_interpolation_targets(
        x,
        target_time,
        extrapolate_days=extrapolate_days,
    )
    out[~valid] = np.nan

    return out, valid


def copy_solution_shell(solution):
    """Copy a {'grid', 'mascons'} solution shell for time/EWH replacement."""
    grid = copy.copy(solution["grid"])
    mascons = {key: value for key, value in solution["mascons"].items()}
    return {"grid": grid, "mascons": mascons}


def align_solution_to_paired_epochs(
    solution,
    pair=None,
    source_epoch_index=None,
    target_time=None,
    target_epoch_index=None,
    interpolate_mask=None,
    extrapolate_days=0.0,
    retain_source_time_bounds=True,
):
    """Return solution on a paired target time axis.

    Parameters
    ----------
    solution : dict
        Mapping with ``grid`` and ``mascons`` entries.
    pair : pandas.DataFrame-like, optional
        Pairing table. By default this expects columns named
        ``centre_epoch_index``, ``jpl_time_days``, ``jpl_epoch_index`` and
        ``aligned_with_jpl``.
    source_epoch_index, target_time, target_epoch_index, interpolate_mask : array-like, optional
        Explicit arrays to use instead of reading the default columns from
        ``pair``.
    extrapolate_days : float, optional
        Number of days beyond the source time range where linear extrapolation
        is allowed. Targets outside this margin are set to NaN.
    retain_source_time_bounds : bool, optional
        If True, output time bounds are copied from the paired source epochs.

    Returns
    -------
    out : dict
        Solution with the same top-level structure as ``solution``.
    metadata : dict
        Source/target indices and interpolation flags.
    """
    out = copy_solution_shell(solution)
    source_time = np.asarray(solution["grid"].time, dtype=float)
    source_ewh = np.asarray(solution["mascons"]["ewh"], dtype=np.float32)

    if pair is None and source_epoch_index is None and target_time is None:
        out["grid"].time = source_time.copy()
        out["grid"].decyear = days_to_decyear(out["grid"].time)
        if getattr(solution["grid"], "time_bounds", None) is not None:
            out["grid"].time_bounds = np.asarray(solution["grid"].time_bounds, dtype=float).copy()
        out["mascons"]["ewh"] = source_ewh.copy()

        n_epochs = len(out["grid"].time)
        metadata = {
            "source_epoch_index": np.arange(n_epochs, dtype=int),
            "target_epoch_index": np.arange(n_epochs, dtype=int),
            "interpolated": np.zeros(n_epochs, dtype=bool),
            "within_source_time_range": np.ones(n_epochs, dtype=bool),
        }
        return out, metadata

    if pair is not None:
        if source_epoch_index is None:
            source_epoch_index = pair["centre_epoch_index"].to_numpy(dtype=int)
        if target_time is None:
            target_time = pair["jpl_time_days"].to_numpy(dtype=float)
        if target_epoch_index is None:
            target_epoch_index = pair["jpl_epoch_index"].to_numpy(dtype=int)
        if interpolate_mask is None:
            interpolate_mask = ~pair["aligned_with_jpl"].to_numpy(dtype=bool)

    source_epoch_index = np.asarray(source_epoch_index, dtype=int)
    target_time = np.asarray(target_time, dtype=float)
    if target_epoch_index is None:
        target_epoch_index = np.arange(len(target_time), dtype=int)
    else:
        target_epoch_index = np.asarray(target_epoch_index, dtype=int)
    if interpolate_mask is None:
        interpolate_mask = np.zeros(len(target_time), dtype=bool)
    else:
        interpolate_mask = np.asarray(interpolate_mask, dtype=bool)

    ewh_on_target = source_ewh[source_epoch_index].copy()
    within_source_time_range = np.ones(len(target_time), dtype=bool)

    if interpolate_mask.any():
        interp_ewh, valid = interpolate_mascon_ewh(
            source_time,
            source_ewh,
            target_time[interpolate_mask],
            extrapolate_days=extrapolate_days,
        )
        ewh_on_target[interpolate_mask] = interp_ewh
        within_source_time_range[interpolate_mask] = valid

    out["grid"].time = target_time
    out["grid"].decyear = days_to_decyear(target_time)
    if retain_source_time_bounds and getattr(solution["grid"], "time_bounds", None) is not None:
        out["grid"].time_bounds = np.asarray(solution["grid"].time_bounds, dtype=float)[
            source_epoch_index
        ].copy()
    out["mascons"]["ewh"] = ewh_on_target

    metadata = {
        "source_epoch_index": source_epoch_index,
        "target_epoch_index": target_epoch_index,
        "interpolated": interpolate_mask,
        "within_source_time_range": within_source_time_range,
    }
    return out, metadata
