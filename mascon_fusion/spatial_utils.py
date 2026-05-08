"""
Module: spatial_utils.py

Description:
Utilities for working with spatial data including converting to/from 
lat/lon coordinates to cartesian (x/y/z) and calculating Earth radius
at a specific latitude on the WGS84 ellipsoid.

Functions:
- lonlat_to_cartesian
- cartesian_to_lonlat
- earth_radius

Author: R McGirr 2026-03
"""

import numpy as np

def lonlat_to_cartesian(lon, lat, R=1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat) 

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x, y, z

def cartesian_to_lonlat(x, y, z):
    """
    calculates lon, lat coordinates from cartesian coordinates
    """
    R = np.sqrt(x**2 + y**2 + z**2)
    lat_r = np.arcsin(z/R)
    lon_r = np.arctan2(y, x)
    lon = np.degrees(lon_r)
    lat = np.degrees(lat_r)
    return lon, lat

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
        
    Input
    ---------
    lat: vector or latitudes in degrees  
        
    Output
    ----------
    r: vector of radius in meters
        
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
        
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
        / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r
