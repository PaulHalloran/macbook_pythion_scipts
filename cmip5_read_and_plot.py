from mpl_toolkits.basemap import Basemap, cm
# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os
import subprocess

potential_files = os.system('/project/champ/mec/bin/managecmip list-local -e rcp85 -f mon -v tas')
print potential_files
print 'hello'
