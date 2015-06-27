import numpy
from matplotlib.pyplot import *
from iris import *
from iris.analysis import *
import iris.quickplot as quickplot

directory = '/home/ph290/Documents/teaching/'

salinity_file_1 = 'salinity_annual_5deg.nc'
salinity_file_2 = 'salinity_annual_1deg.nc'
temperature_file_1 = 'temperature_annual_5deg.nc'
temperature_file_2 = 'temperature_annual_1deg.nc'
silicate_file_1 = 'silicate_annual_5deg.nc'
silicate_file_2 = 'silicate_annual_1deg.nc'
phosphate_file_1 = 'phosphate_annual_5deg.nc'
phosphate_file_2 = 'phosphate_annual_1deg.nc'  
nitrate_file_1 = 'nitrate_annual_1deg.nc'     
nitrate_file_2 = 'nitrate_annual_5deg.nc'
oxygen_file_1 = 'oxygen_saturation_annual_1deg.nc'  
oxygen_file_2 = 'oxygen_saturation_annual_5deg.nc'  

temporary_cubes = load(directory+salinity_file_1)
#note here that we are making the code more readable by joining together two strings using the '+' symbol 

print temporary_cubes

salinity_cube =  load_cube(directory+salinity_file_1,'Statistical Mean')
salinity_cube = salinity_cube[0] 

#and going through a similar process we can read in the other files 


import seawater
#This is the module that will calculate the seawaters density for us 

temperature_cube =  load_cube(directory+temperature_file_1,'Statistical Mean')
temperature_cube = temperature_cube[0] 

salinity_cube =  load_cube(directory+salinity_file_1,'Statistical Mean')
salinity_cube = salinity_cube[0] 

density_cube = temperature_cube.copy()
#first we need to make a new cube to hold the density data when we calculate it. This is what we do here. 

density_cube.data = seawater.dens(temperature_cube.data,salinity_cube.data,1)
#This script works with the data within a cube - i.e. it does not know what to do with the metadata, so we have to specify that we're only dealing with the data part (by using '.data') 

#We can now treat this density cube as we would any other cube

#density_cube.coord('depth').points=density_cube.coord('depth').points*(-1.0)
meridional_slice = density_cube.extract(Constraint(longitude=182.5))
quickplot.contourf(meridional_slice, 30, coords=['latitude','depth']) 
show() 


