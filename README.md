# mascon_fusion

`mascon_fusion` is a Python framework for generating fused GRACE/GRACE-FO mascon solutions from multiple independent mascon products within a common inversion framework.

This software combines mascon solutions from multiple analysis centers after coastline harmonization using coastline resolution improvement (CRI) filtering techniques (Wiese et al., 2016, Water Resources Research).

The COM-TG mascon product created using this software package is available for download at [ANU data commons](https://dx.doi.org/10.25911/61dceb9c03396). 

---

## Citation

Please cite McGirr and Tregoning, 2026 (*A combined mascon product from multiple analyses of GRACE and GRACE Follow-On space gravity data*, GRL) when using the mascon_fusion software. 

---

## Requirements
- Python >= 3.10
- numpy
- scipy
- netCDF4
- datetime
- matplotlib
- cartopy

---

## Repository structure
```text
mascon_fusion/
│   └── mascon_fusion/
│       ├── diagnostics.py      # Diagnostic routines for analyzing fusion
│       ├── fusion.py           # Main fusion routines
│       ├── grid.py             # Class for working with gridded data   
│       ├── io.py               # Input/output utilities
│       ├── preprocessing.py    # Utilities for preparing observations for fusion
│       ├── spatial_utils.py    # Spatial transform utilities
│       └── time_utils.py       # Time-conversion helpers
├── example_usuage.ipynb 
└── README.md 
```

---

## Input Data
The fusion software expects GRACE/GRACE-FO mascon netCDF's in a particular format, whereby the solution is stored on individual mascons in a solution group. This data can be mapped to a grid using the grid mascon_ID. Vectorized mascon information is stored in the mascon_info group. See below file structure.

All GRACE/GRACE-FO mascon netCDF's used in the fusion need to be mapped to a common grid prior.

```text
input_mascons.nc
│
├── lon(lon)
├── lat(lat)
├── mascon_ID(lat, lon)
├── mask(lat, lon)
├── time(time)
│
├── solution/
│   └── lwe_thickness(time, mascon)
│
└── mascon_info/
    ├── id(mascon)
    ├── lat_center(mascon)
    ├── lon_center(mascon)
    └── area(mascon)
```

---

## Output Data
The fused mascon solutions are in NetCDF format using the same file structure as the input data with some additional information on the solution uncertainty written to the solution group. Optionally, the residuals may also be wriiten to a NetCDF using 
