"""
Module: grid.py

Description:
Defines the `regular_grid' class for creating, reading, and manipulating regular
spatial grids. Supports grid construction from bounds, ingestion of netCDF
datasets, coordinate transformations, area calculations on the WGS84 ellipsoid,
and nearest-neighbour mapping between grids.

Functions:
- create_grid
- read_grid
- get_latlon_bounds
- get_latlon_coords
- get_XY_coords
- get_grid_area
- get_mapping_sht
- get_mapping
- nearest_ids

Author: R McGirr 2026-03
"""

import numpy as np
from netCDF4 import Dataset
from .spatial_utils import lonlat_to_cartesian

class regular_grid:
    def __init__(self, fname=None, verbose=False):
        self.fname = fname
        if fname is not None:
            self.read_grid(verbose=verbose)
    
    def create_grid(self, minlat, maxlat, minlon, maxlon, d):
        '''
        create a regular grid given min/max lat,
        min/max lon and grid spaceing
        '''
        print(f'creating grid with {d} degree spacing')
        print(f'min/max lat = {minlat}/{maxlat}, min/max lon = {minlon}/{maxlon}')
        
        # grid spacing
        self.d = d
        self.nlat = int((maxlat-minlat)/d)
        self.nlon = int((maxlon-minlon)/d)
        # grid bounds
        self.lat_b = np.linspace(minlat, maxlat, self.nlat + 1)
        self.lon_b = np.linspace(minlon, maxlon, self.nlon + 1)
	    # grid centroids
        self.lat = np.linspace(minlat+d/2, maxlat-d/2, self.nlat)
        self.lon = np.linspace(minlon+d/2, maxlon-d/2, self.nlon)
	    # create meshgrid bounds
        self.lon_bv, self.lat_bv = np.meshgrid(self.lon_b, self.lat_b)
	    # create meshgrid centroids
        self.lonv, self.latv = np.meshgrid(self.lon, self.lat)
	    # get grid cell areas
        self.get_grid_area()

    def read_grid(self, verbose=True):
        '''
        Read a netCDF file containing lat, lon, time, and
        data variables and store them in dictionaries.
        '''
        with Dataset(self.fname) as ds:
            print('Reading file:', self.fname)
            try:
                self.summary = ds.summary
                print('Summary:')
                print(self.summary)
            except:
                print('No summary found')
            self.var = {}
            self.units = {}
            if verbose: print('\nReading dimensions...')
            for dim in ds.dimensions.keys():
                if verbose: print("%s(%i)"%(ds.dimensions[dim].name,ds.dimensions[dim].size))
            if verbose: print('\nStoring variables in dictionary...')
            for var in ds.variables.keys():
                try:
                    if verbose: print('%s%s: %s'%(var, str(ds[var].shape), str(ds[var].units)))
                    self.units[var] = ds[var].units
                except AttributeError:
                    try: 
                        if verbose: print('%s%s: %s'%(var, str(ds[var].shape), str(ds[var].Units)))
                        self.units[var]= ds[var].Units
                    except AttributeError:
                        if verbose: print('%s%s: units not found'%(var, str(ds[var].shape)))
                        self.units[var] = 'dimensionless'
                self.var[var] = ds[var][:]
            # if groups are present store them in a dictionary
            if len(ds.groups.keys()) > 0:
                for group in ds.groups.keys():
                    self.var[group] = {}
                    for var in ds.groups[group].variables.keys():
                        self.var[group][var] = ds.groups[group].variables[var][:]

            # assume there are lat, lon and time variables
            if 'lat' in self.var.keys() and 'lon' in self.var.keys():
                if len(ds['lat'].shape) == 1 and len(ds['lon'].shape) == 1:
                    self.lat, self.lon = self.var['lat'][:], self.var['lon'][:]
                    self.nlat, self.nlon = len(self.lat), len(self.lon)
                    self.lonv, self.latv = np.meshgrid(self.lon, self.lat)

                    # find delta
                    dlat = np.abs(np.unique(np.diff(self.lat)))
                    dlon = np.abs(np.unique(np.diff(self.lon)))
                    if dlat.size != 1 or dlon.size != 1 or dlat != dlon:
                        raise ValueError('Supplied grid is not regular')
                    self.d = np.abs(self.lat[1] - self.lat[0])
                    self.get_latlon_bounds()
                    self.get_grid_area()
                elif len(ds['lat'].shape) == 2 and len(ds['lon'].shape) == 2:
                    self.lat, self.lon = self.var['lat'][:,:], self.var['lon'][:,:]
                self.extent = [np.min(self.lon),np.max(self.lon),np.min(self.lat),np.max(self.lat)]
                if verbose: print('\nExtent of grid is ',self.extent)
            
            # if x, y variables are present, convert to lat/lon
            elif 'x' in self.var.keys() and 'y' in self.var.keys():
                if len(ds['x'].shape) == 1 and len(ds['y'].shape) == 1:
                    self.x, self.y = self.var['x'][:], self.var['y'][:]
                    self.xv, self.yv = np.meshgrid(self.x, self.y, indexing='ij')
                    dx = np.abs(np.unique(np.diff(self.x)))
                    dy = np.abs(np.unique(np.diff(self.y)))
                    if dx.size != 1 or dy.size != 1 or dx != dy:
                        raise ValueError('Supplied grid is not regular')
                    else:
                        self.delta = np.abs(self.x[1] - self.x[0])
                        if verbose: print('\nCalculating area of regular grid...')
                        if verbose: print('Grid spacing is %.2f m'%(self.delta))
                        self.area = np.ones_like(self.x) * self.delta**2
                elif len(ds['x'].shape) == 2 and len(ds['y'].shape) == 2:
                    self.xv, self.yv = self.var['x'][:,:], self.var['y'][:,:]
                if verbose: print('\nExtent of grid is ',[np.min(self.xv),np.max(self.xv),np.min(self.yv),np.max(self.yv)])
            else:
                print('lat/lon variables not found!')

    def get_latlon_bounds(self):
        '''
        get lat/lon bounds assuming lat/lon exists
        '''
    	# lats first
        self.lat_b = np.zeros(self.nlat + 1)
        self.lat_b[:self.nlat] = self.lat - self.d/2
        self.lat_b[-1] = self.lat[-1] + self.d/2
        # then lons
        self.lon_b = np.zeros(self.nlon + 1)
        self.lon_b[:self.nlon] = self.lon - self.d/2
        self.lon_b[-1] = self.lon[-1] + self.d/2
	    # create meshgrid
        self.lon_bv, self.lat_bv = np.meshgrid(self.lon_b, self.lat_b)

    def get_latlon_coords(self,xy_crs='epsg:4978',geo_crs='epsg:4326'):
        '''
        Convert xy coordinates to lat/lon coordinates
        '''
        from pyproj import Transformer
        xy_to_geo = Transformer.from_crs(xy_crs, geo_crs)
        self.latv, self.lonv = xy_to_geo.transform(self.xv, self.yv)

    def get_XY_coords(self,xy_crs='epsg:4978',geo_crs='epsg:4326'):
        '''
        Convert lat/lon coordinates to xy coordinates
        '''
        from pyproj import Transformer  # library to allow coordinate transforms
        geo_to_xy = Transformer.from_crs(geo_crs, xy_crs)
        self.xv, self.yv = geo_to_xy.transform(self.latv,self.lonv)   

    def get_grid_area(self, verbose=True):
        '''
        Calculate area of each cell assuming grid
        is on the WGS84 ellipsoid.
        '''
        from pyproj import Geod
        from shapely.geometry import Polygon

        geod = Geod(ellps="WGS84")
        lat_areas = np.zeros(self.nlat)
        for i in range(self.nlat):
            # create a polygon around the grid
            poly_bnds = np.array([[0,self.d,self.d,0],[self.lat_b[i],self.lat_b[i],self.lat_b[i+1],self.lat_b[i+1]]]).T
            lat_areas[i],_ = geod.geometry_area_perimeter(Polygon(poly_bnds))
        # create the area mesh
        self.area = np.tile(lat_areas, (self.nlon, 1)).T
        
        grid_area_km2 = np.nansum(self.area)/1e6
        print(f'Surface area of grid is {grid_area_km2:.2f} km^2')
    

    def get_mapping_sht(self, d=0.5, llon=0., rlon=360., llat=90., rlat=-90., inv=True):
        '''
        Creates a regular grid of points on the surface of the Earth
        with a spacing of d degrees.  Then finds the nearest
        mascon to each grid point.  The result is a mapping from the
        grid points to the mascons.
        '''
        from scipy.spatial import cKDTree
        # create new lats/lons
        if inv:
            lat = np.r_[llat:rlat:-1 * d]
        else:
            lat = np.r_[llat:rlat:d]
        lon = np.r_[llon:rlon:d]
        lon2d, lat2d = np.meshgrid(lon, lat)
        xG, yG, zG = lonlat_to_cartesian(lon2d.flatten(), lat2d.flatten())
        xT, yT, zT = lonlat_to_cartesian(self.lonv.flatten(), self.latv.flatten())
        # map frpm T coords to G coords
        tree = cKDTree(list(zip(xT, yT, zT)))
        d, inds = tree.query(list(zip(xG, yG, zG)), k=1)
        # replace lon/lat with new values
        self.lon, self.lat = lon, lat
        self.lonv, self.latv = lon2d, lat2d
        self.idx_near = np.int_(inds.reshape(lon2d.shape))

    def get_mapping(self, to_grid):
        '''
        Get mapping from one grid to another
        using kd-tree for nearest neighbour.
        '''       
        from scipy.spatial import cKDTree
        xG, yG, zG = lonlat_to_cartesian(to_grid.lonv.flatten(), to_grid.latv.flatten())
        xT, yT, zT = lonlat_to_cartesian(self.lonv.flatten(), self.latv.flatten())
        tree = cKDTree(list(zip(xT, yT, zT)))
        d, inds = tree.query(list(zip(xG, yG, zG)), k=1)
        self.idx_near = np.int_(inds.reshape(to_grid.lonv.shape))

    def nearest_idx(self, lat, lon):
        """
        Find the nearest grid cell to a given lat, lon
        """
        # make sure lons are in the same range
        if np.any(self.lonv < 0) and lon > 180:
            lon -= 360
        elif np.any(self.lonv > 180) and lon < 0:
            lon += 360
        dist = np.sqrt((self.latv - lat)**2 + (self.lonv - lon)**2)
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        return idx
    