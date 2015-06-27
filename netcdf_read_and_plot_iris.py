'''
Script to plot out the surface ocean variables from any cmip5 model and any run (historical, RCP etc.) (except a few). Currently just plots the first (i.e. earliest field). NOTE - may need to add exceptions for other models when plotting different variables, not yet sure
'''

from mpl_toolkits.basemap import Basemap, cm
from netCDF4 import Dataset as NetCDFFile
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import numpy.ma as ma
import iris
import iris.quickplot as qplt
import iris.analysis.cartography

def read_and_plot(file_name,temp_model,var_name,output_directory):
#function to read ans plot CMIP5 data

    print file_name.strip()

    var=iris.load_cube(file_name.strip())
 
    # Projection is required because the coords are not monotonic.  
    var_regridded, extent = iris.analysis.cartography.project(var[0,:,:], ccrs.PlateCarree())

    # We now have two recognisable "x" and "y" coordinates,
    # which currently confuses plotting code.
    var_regridded.remove_coord("latitude")
    var_regridded.remove_coord("longitude")

    # set plot window size
    qplt.plt.figure()

    #plot
    qplt.contourf(var_regridded,30)
 
    # add coastlines
    qplt.plt.gca().coastlines()
    plt.title(temp_model+' '+var_name)
    # show the plot
    #qplt.plt.show()
    outfile = output_directory+'/'+temp_model.strip()+'_'+var_name+'.png'
    plt.savefig(outfile)

  
'''
Main code
'''

######################################################
######################################################
run='historical'
var_name='tos'
output_directory='/net/home/h04/hador/temp_plots'
######################################################
######################################################

#Read in the names of all of the CMIP5 files with the variable 'fgco2'
file_data=[]
p = Popen("/project/champ/mec/bin/managecmip list-local -e "+run+" -f mon -v "+var_name,stdout=PIPE,stderr=PIPE,shell=True)

for line in p.stdout:
    file_data.append(line.split('/'))
for err in p.stderr:
    pass

#split the long file-names into sections to pull out model names, variable names etc.
model_centers = []
model= []
run_name= []
variable= [] 

for temp in file_data:
    model_centers.append(temp[6])
    model.append(temp[7])
    run_name.append(temp[8])
    variable.append(temp[14])  

#come up with a unique list of models
unique_model=np.unique(np.array(model))

#lind the first items in the list of all the model data files relating to each separate model
model_loc_base=[]
for temp_model in unique_model:
    model_loc=[]
    for i,x in enumerate(model):
        if x == temp_model:
            model_loc.append(i)
    model_loc_base.append(model_loc)
    
#read in and plot the model data
for i,temp_model in enumerate(unique_model):
    #if i < 10:
    #    continue
    #just used in testing to skip past fits models
    temp=model_loc_base[i]     
    file_name = '/'.join(file_data[temp[0]])
################################################################
###        WHY DO THESE MODELS FAIL TO PLOT?                ####
################################################################
#currently getting funny error with GFDL, but that I think is because the uploaded poor-netcdfs.
#more of a problem is the error:
#ValueError: Calling _get_lat_lon_coords() with multiple lat or lon coords is currently disallowed
#what does this mean?
    test=-1
    if file_name.find('GFDL-CM2p1') != -1: test = +1
    if file_name.find('GFDL-CM3') != -1: test = +1
    if file_name.find('GFDL-ESM2M') != -1: test = +1
    #if file_name.find('MIROC5') != -1: test = +1
    if file_name.find('MRI-CGCM3') != -1: test = +1
    if file_name.find('BCC') != -1: test = +1
    if file_name.find('inmcm4') != -1: test = +1
    if test == -1:
        read_and_plot(file_name,temp_model,var_name,output_directory)

