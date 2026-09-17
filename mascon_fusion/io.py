"""
Module: io.py

Description:
Input/output utilities for reading/writing mascon fusion files
and CRI mascon grids.

Functions:
- write_fusion_netcdf
- load_mascon_fusion_grid
- load_mascon_grid
- write_fusion_residuals_netcdf
- load_residuals

Author: R McGirr 2026-03
"""

import os
import datetime
import netCDF4 as nc
from weakref import ref
import numpy as np
from netCDF4 import Dataset
from .grid import regular_grid
from .time_utils import get_decyear, decyear_to_daysince

def load_mascon_fusion_grid(filename, load_solution=False, remove_nans=False):
    """
    Load full global mascon grid and metadata from NetCDF file.

    Parameters
    ----------
    filename : str
        Path to mascon NetCDF file
    load_solution : bool, optional
        If True, load EWH solution and time

    Returns
    -------
    grid : object
        Contains lat, lon, latv, lonv, area, mask, mascon_id, and optionally ewh
    mascons : dict
        Contains mascon metadata (id, lat, lon, area, optionally ewh)
    """

    # Load full grid
    grid = regular_grid(filename)

    # No subsetting — just reshape variables to global grid
    grid.latv = grid.latv
    grid.lonv = grid.lonv
    grid.area = grid.area
    grid.mask = np.array(grid.var['mask'], dtype=bool)
    grid.mascon_id = grid.var['mascon_ID']
    

    # Load mascon metadata
    mascons = {
        'id': grid.var['mascon_info']['id'],
        'lat': grid.var['mascon_info']['lat_center'],
        'lon': grid.var['mascon_info']['lon_center'],
        'area': grid.var['mascon_info']['area']
    }

    # create nmascons 
    grid.nmascons = len(mascons['id'])

    
    # Load solution if requested
    if load_solution:

        # Convert from cm to meters
        mascons['ewh'] = grid.var['solution']['lwe_thickness'] / 100
        mascons['SE'] = grid.var['solution']['SE'] / 100

        grid.time = grid.var['time']
        try:
            grid.time_bounds = grid.var['solution']['time_bounds']
        except:
            print("Warning: time_bounds not found in solution group; setting to None")
            grid.time_bounds = None
        grid.decyear = get_decyear(grid.time)

        # Build gridded EWH
        grid.ewh = mascons['ewh'][:, grid.mascon_id - 1].reshape(
                   len(grid.time), grid.nlat, grid.nlon
        ).data

        # Build gridded SE
        grid.SE = mascons['SE'][:, grid.mascon_id - 1].reshape(
                   len(grid.time), grid.nlat, grid.nlon
        ).data

        grid.sigma0 = grid.var['solution']['sigma0']
        grid.block_rms = grid.var['solution']['block_rms'] / 100

    if remove_nans:
        # some epochs are all nan where there was no solution
        nan_mask = np.isnan(grid.ewh[:,0,0])

        # mask anything with a time dimension
        grid.decyear = grid.decyear[~nan_mask]
        grid.ewh = grid.ewh[~nan_mask]
        grid.SE = grid.SE[~nan_mask]
        grid.sigma0 = grid.sigma0[~nan_mask]
        grid.block_rms = grid.block_rms[~nan_mask]

        # mascon info items
        mascons['ewh'] = mascons['ewh'][~nan_mask]
        mascons['SE'] = mascons['SE'][~nan_mask]

    # Clean up raw NetCDF variable container
    del grid.var

    return grid, mascons

def write_fusion_netcdf(
    filename, 
    grid, 
    mascons,
    experiment_name,
    block_scalars,
    area_weighting,
    block_normalised,
):
    """
    Write fusion results to NetCDF file.

    Parameters
    ----------
    filename : str
        Output NetCDF filename

    grid : regular_grid
        Must contain:
            'lat'
            'lon'
            'decyear'
            'mascon_id'
            'area'
            'mask'

    mascons : dict
        Must contain:
            'lat'
            'lon'
            'area'
            'id'
            'ewh'
            'SE'
            'sigma0'
            'block_rms

    experiment_name : str

    block_scalars : dict

    area_weighting : bool

    block_normalised : bool
    """
    if os.path.exists(filename): os.remove(filename)

    nmascons = len(mascons["id"])
    ntimes = len(grid.decyear)
    products = list(mascons['block_rms'].keys())
    nproducts = len(products)

    nc = Dataset(filename, "w", format="NETCDF4")

    # --------------------------------------------------
    # Dimensions
    # --------------------------------------------------

    nc.createDimension("time", ntimes)
    nc.createDimension("mascon", nmascons)
    nc.createDimension("product", nproducts)
    nc.createDimension('lon', len(grid.lon))
    nc.createDimension('lat', len(grid.lat))

    # --------------------------------------------------
    # Grid coordinates
    # --------------------------------------------------

    time_var = nc.createVariable('time', 'i4', ('time'))
    ref_date = datetime.date(2002, 1, 1)
    time_var.units = f'days since {ref_date.strftime("%Y-%m-%d")}T00:00:00Z'
    time_var[:] = decyear_to_daysince(grid.decyear, ref_date)

    lat_var = nc.createVariable('lat', 'f8', ('lat',))
    lat_var.units = "degrees_north"
    lat_var[:] = grid.lat

    lon_var = nc.createVariable('lon', 'f8', ('lon',))
    lon_var.units = "degrees_east"
    lon_var[:] = grid.lon

    area_var = nc.createVariable("area", "f8", ('lat','lon'))
    area_var.units = "m^2"
    area_var[:] = grid.area

    mask_var = nc.createVariable("mask", "i4", ('lat','lon'))
    mask_var.units = "dimensionless"
    mask_var[:] = grid.mask

    id_var = nc.createVariable("mascon_ID", "i4", ('lat','lon'))
    id_var.units = "dimensionless"
    id_var[:] = grid.mascon_id

    product_var = nc.createVariable("product", str, ('product',))
    product_var[:] = np.array(products, dtype='S')

    # --------------------------------------------------
    # Mascon group 
    # --------------------------------------------------

    mascon_group = nc.createGroup('mascon_info')

    mascon_id_var = mascon_group.createVariable('id', 'i4', ('mascon',))
    mascon_id_var[:] = mascons["id"]

    mascon_lat_var = mascon_group.createVariable('lat_center', 'f8', ('mascon',))
    mascon_lat_var.units = "degrees_north"
    mascon_lat_var[:] = mascons["lat"]

    mascon_lon_var = mascon_group.createVariable('lon_center', 'f8', ('mascon',))
    mascon_lon_var.units = "degrees_east"
    mascon_lon_var[:] = mascons["lon"]

    mascon_area_var = mascon_group.createVariable('area', 'f8', ('mascon',))
    mascon_area_var.units = "m^2"
    mascon_area_var[:] = mascons["area"] 

    # --------------------------------------------------
    # Solution group 
    # --------------------------------------------------

    solution_group = nc.createGroup('solution')    

    ewh_var = solution_group.createVariable('lwe_thickness', 'f4', ('time','mascon'))
    ewh_var.units = "cm"
    ewh_var.long_name = "fused equivalent water height"
    ewh_var[:] = mascons['ewh']*100

    se_var = solution_group.createVariable("SE", "f4", ("time", "mascon"))
    se_var.units = "cm"
    se_var.long_name = "standard error"
    se_var[:] = mascons["SE"]*100

    sigma_var = solution_group.createVariable("sigma0", "f4", ("time",))
    sigma_var.long_name = "variance factor"
    sigma_var[:] = mascons["sigma0"]

    rms_var = solution_group.createVariable("block_rms", "f4", ("time", "product"))
    rms_var.units = "cm"
    rms_var.long_name = "block RMS residual"

    rms_array = np.zeros((ntimes, nproducts))

    for i, name in enumerate(products):
        rms_array[:, i] = mascons['block_rms'][name]

    rms_var[:] = rms_array*100

    # --------------------------------------------------
    # Global attributes
    # --------------------------------------------------

    nc.title = "GRACE mascon fusion solution"
    nc.experiment_name = experiment_name
    nc.area_weighting = str(area_weighting)
    nc.block_normalised = str(block_normalised)
    nc.block_scalars = ", ".join(f"{k}={v}" for k,v in block_scalars.items())

    nc.history = f"Created {datetime.datetime.utcnow().isoformat()} UTC"
    nc.source = "Mascon fusion weighted least squares inversion"

    nc.close()

    print(f"Wrote {filename}")

def write_fusion_residuals_netcdf(
    filename,
    grid,
    mascons,
    insols,
    experiment_name,
):
    """
    Write fusion residuals to NetCDF file.

    Parameters
    ----------
    filename : str
        Output NetCDF filename

    grid : regular_grid
        Target grid object (must contain lat, lon, decyear)

    mascons : dict
        Must contain:
            mascons["residuals"][product] -> ndarray (ntimes, nmascons_product)

    insols : dict
        Input product grid objects; used for product-specific mascon_id grids

    experiment_name : str
        Name of experiment to store as metadata
    """
    if os.path.exists(filename):
        os.remove(filename)

    products = list(mascons["residuals"].keys())
    ntimes = len(grid.decyear)

    nc = Dataset(filename, "w", format="NETCDF4")

    # --------------------------------------------------
    # Dimensions
    # --------------------------------------------------
    nc.createDimension("time", ntimes)
    nc.createDimension("lat", len(grid.lat))
    nc.createDimension("lon", len(grid.lon))

    # create a dim for each input mascon too
    for name in insols.keys():
        nc.createDimension(f"mascon_{name}", insols[name].nmascons)

    # --------------------------------------------------
    # Grid coordinates
    # --------------------------------------------------
    time_var = nc.createVariable("time", "i4", ("time",))
    ref_date = datetime.date(2002, 1, 1)
    time_var.units = f'days since {ref_date.strftime("%Y-%m-%d")}T00:00:00Z'
    time_var[:] = decyear_to_daysince(grid.decyear, ref_date)

    lat_var = nc.createVariable("lat", "f8", ("lat",))
    lat_var.units = "degrees_north"
    lat_var[:] = grid.lat

    lon_var = nc.createVariable("lon", "f8", ("lon",))
    lon_var.units = "degrees_east"
    lon_var[:] = grid.lon

    area_var = nc.createVariable("area", "f8", ('lat','lon'))
    area_var.units = "m^2"
    area_var[:] = grid.area

    mask_var = nc.createVariable("mask", "i4", ('lat','lon'))
    mask_var.units = "dimensionless"
    mask_var[:] = grid.mask

    # --------------------------------------------------
    # Mascon id grid
    # --------------------------------------------------
    id_var = {}
    for name in products:
        id_var[name] = nc.createVariable(f"mascon_ID_{name}", "i4", ('lat','lon'))
        id_var[name].units = "dimensionless"
        id_var[name].long_name = f"{name} mascon ID on target grid"
        id_var[name][:] = insols[name].mascon_id

    # --------------------------------------------------
    # Residuals group
    # --------------------------------------------------
    res_group = nc.createGroup("residuals")

    for name in products:
        # residual vector on product mascons
        res_var = res_group.createVariable(
            f"{name}", "f4",
            ("time", f"mascon_{name}")
        )
        res_var.units = "cm"
        res_var.long_name = f"{name} residuals on native mascons"
        res_var[:] = mascons["residuals"][name] * 100.0

    # --------------------------------------------------
    # Global attributes
    # --------------------------------------------------
    nc.title = "GRACE mascon fusion residuals"
    nc.experiment_name = experiment_name
    nc.products = ", ".join(products)
    nc.history = f"Created {datetime.datetime.utcnow().isoformat()} UTC"
    nc.source = "Mascon fusion weighted least squares inversion"

    nc.close()

    print(f"Wrote {filename}")

def load_mascon_grid(filename, load_solution=False):
    """
    Load full global mascon grid and metadata from NetCDF file.

    Parameters
    ----------
    filename : str
        Path to mascon NetCDF file
    load_solution : bool, optional
        If True, load EWH solution and time

    Returns
    -------
    grid : object
        Contains lat, lon, latv, lonv, area, mask, mascon_id, and optionally ewh
    mascons : dict
        Contains mascon metadata (id, lat, lon, area, optionally ewh)
    """

    # Load full grid
    grid = regular_grid(filename)

    # No subsetting — just reshape variables to global grid
    grid.latv = grid.latv
    grid.lonv = grid.lonv
    grid.area = grid.area
    grid.mask = np.array(grid.var['mask'],dtype=bool)
    grid.mascon_id = grid.var['mascon_ID']
    

    # Load mascon metadata
    mascons = {
        'id': grid.var['mascon_info']['id'],
        'lat': grid.var['mascon_info']['lat_center'],
        'lon': grid.var['mascon_info']['lon_center'],
        'area': grid.var['mascon_info']['area']
    }

    # create nmascons 
    grid.nmascons = len(mascons['id'])
    
    # Load solution if requested
    if load_solution:

        # Convert from cm to meters
        mascons['ewh'] = grid.var['solution']['lwe_thickness'] / 100

        grid.time = grid.var['solution'].get('time', grid.var.get('time'))
        try:
            grid.time_bounds = grid.var['solution']['time_bounds']
        except:
            print("Warning: time_bounds not found in solution group; setting to None")
            grid.time_bounds = None
        grid.decyear = get_decyear(grid.time)

        # Build gridded EWH
        grid.ewh = mascons['ewh'][:, grid.mascon_id - 1].reshape(
                   len(grid.time), grid.nlat, grid.nlon
        )

    # Clean up raw NetCDF variable container
    del grid.var

    return grid, mascons

def write_mascon_grid_global(filename, grid, mascons, mask, title=None, summary=None):

    # check if file exists
    if os.path.exists(filename): os.remove(filename)

    # convert ewh in m to cm
    lwe_thickness = mascons['ewh']*100

    # Now write mascon_placement to a netcdf file
    nc_file = Dataset(filename, 'w', format='NETCDF4')
    nc_file.set_auto_mask(False)

    # create dimensions
    lon_dim = nc_file.createDimension('lon', len(grid.lon))
    lat_dim = nc_file.createDimension('lat', len(grid.lat))
    time_dim = nc_file.createDimension('time', len(grid.time))
    bounds_dim = nc_file.createDimension('bounds', 2)
    mascon_dim = nc_file.createDimension('mascon', int(np.max(grid.mascon_id)))

    # Create variables
    grid_lon_var = nc_file.createVariable('lon', 'f8', ('lon',))
    grid_lat_var = nc_file.createVariable('lat', 'f8', ('lat',))
    grid_mascon_ID_var = nc_file.createVariable('mascon_ID', 'i4', ('lat','lon'))
    grid_mask_var = nc_file.createVariable('mask', 'i4', ('lat','lon'))

    # create group for mascon info
    mascon_group = nc_file.createGroup('mascon_info')
    mascon_ID_var = mascon_group.createVariable('id', 'i4', ('mascon',))
    mascon_lat_var = mascon_group.createVariable('lat_center', 'f8', ('mascon',))
    mascon_lon_var = mascon_group.createVariable('lon_center', 'f8', ('mascon',))
    mascon_area_var = mascon_group.createVariable('area', 'f8', ('mascon',))

    # create group for solution
    solution_group = nc_file.createGroup('solution')
    time_var = solution_group.createVariable('time', 'f8', ('time'))
    ewh_var = solution_group.createVariable('lwe_thickness', 'f8', ('time','mascon'))
    time_bounds_var = solution_group.createVariable('time_bounds', 'f8', ('time', 'bounds'))

    # write to grid variables
    grid_lon_var[:] = grid.lon
    grid_lat_var[:] = grid.lat
    grid_mascon_ID_var[:] = grid.mascon_id
    grid_mask_var[:] = mask

    # write to mascon info variables
    mascon_ID_var[:] = mascons['id']
    mascon_lat_var[:] = mascons['lat']
    mascon_lon_var[:] = mascons['lon']
    mascon_area_var[:] = mascons['area']

    # write to solution variables
    time_var[:] = grid.time
    ewh_var[:] = lwe_thickness
    time_bounds_var[:] = grid.time_bounds

    # lon_var
    for lon_var in [grid_lon_var, mascon_lon_var]:
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'longitude'
        lon_var.standard_name = 'longitude'
        lon_var.axis = 'X'
        lon_var.valid_min = lon_var[:].min()
        lon_var.valid_max = lon_var[:].max()

    # lat_var
    for lat_var in [grid_lat_var, mascon_lat_var]:
        lat_var.units = 'degrees_north'
        lat_var.long_name = 'latitude'
        lat_var.standard_name = 'latitude'
        lat_var.axis = 'Y'
        lat_var.valid_min = lat_var[:].min()
        lat_var.valid_max = lat_var[:].max()

    # time_var
    time_var.units = 'days since 2002-01-01T00:00:00Z'
    time_var.long_name = 'time'
    time_var.standard_name = 'time'
    time_var.axis = 'T'
    time_var.calendar = 'gregorian'
    time_var.bounds = 'time_bounds'

    # time_bounds_var
    time_bounds_var.long_name = 'time boundaries'
    time_bounds_var.units = 'days since 2002-01-01T00:00:00Z'
    time_bounds_var.comment = 'time bounds for the beginning and end of each month'

    # mask_var
    grid_mask_var.units = 'dimensionless'
    grid_mask_var.long_name = 'Land mask'
    grid_mask_var.standard_name = 'mask'
    grid_mask_var.coordinates = 'lat lon'
    grid_mask_var.valid_min = mask.min()
    grid_mask_var.valid_max = mask.max()
    grid_mask_var.description = 'land mask for the grid'

    # mascon_ID_var
    for mascon_ID_var in [grid_mascon_ID_var, mascon_ID_var]:
        mascon_ID_var.units = 'dimensionless'
        mascon_ID_var.long_name = 'Mascon_Identifier'
        mascon_ID_var.standard_name = 'Mascon_ID'
        mascon_ID_var.coordinates = 'lat lon'
        mascon_ID_var.valid_min = grid.mascon_id.min()
        mascon_ID_var.valid_max = grid.mascon_id.max()
        mascon_ID_var.description = 'Primary mascon number mapped to the grid'

    # area_var
    mascon_area_var.units = 'm2'
    mascon_area_var.long_name = 'mascon area'
    mascon_area_var.standard_name = 'mascon area'
    mascon_area_var.valid_min = mascon_area_var[:].min()
    mascon_area_var.valid_max = mascon_area_var[:].max()

    # ewh_var
    ewh_var.units = 'cm'
    ewh_var.long_name = 'Liquid_Water_Equivalent_Thickness'
    ewh_var.standard_name = 'Liquid_Water_Equivalent_Thickness'
    ewh_var.coordinates = 'time mascons'
    ewh_var.grid_mapping = 'WGS84'
    ewh_var.valid_min = lwe_thickness.min()
    ewh_var.valid_max = lwe_thickness.max()

    if title is not None: nc_file.title = title
    if summary is not None: nc_file.summary = summary

    # Close the NetCDF file
    nc_file.close()

def load_residuals(grid, path):
    print(f"Loading residuals from {path}")
    d = nc.Dataset(path)
    grid_residuals = {}
    residuals = {'ewh': {},
                'mascon_ID': {}
                }
    print(f"Products found: {d.products}")
    centres = d.products.split(', ')
    for ce in centres:
        residuals['ewh'][ce] = d[f'residuals/{ce}'][:, :].data / 100
        residuals['mascon_ID'][ce] = d[f'mascon_ID_{ce}'][:].data
        # reshape the residuals to the same shape as the fusion solution
        grid_residuals[ce] = residuals['ewh'][ce][:, residuals['mascon_ID'][ce] - 1].reshape(
                        d['time'].shape[0], grid.nlat, grid.nlon
        )
    return grid_residuals, residuals
