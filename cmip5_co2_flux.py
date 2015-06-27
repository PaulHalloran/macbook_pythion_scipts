import numpy as np
import matplotlib.pyplot as plt
import iris
import glob
import iris.experimental.concatenate
import iris.analysis
import iris.quickplot as qplt
import iris.analysis.cartography
import cartopy.crs as ccrs
import subprocess
from iris.coords import DimCoord
import iris.coord_categorisation

print 'first remove inmcm4 (fgco2_Omon_inmcm4_rcp85_r1i1p1_200601-210012.nc) from your list of files'

def my_callback(cube, field,files_tmp):
    # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
    cube.attributes.pop('history')
    cube.attributes.pop('tracking_id')
    cube.attributes.pop('creation_date')
    #if np.size(cube) > 1:
    #cube = iris.experimental.concatenate.concatenate(cube)
    return cube

def monthly_to_yearly(cube):
    if np.size(cube._aux_coords_and_dims) < 2:
        iris.coord_categorisation.add_year(cube, 'time', name='year2')
    cube_tmp = cube.aggregated_by('year2', iris.analysis.MEAN)
    return cube_tmp


directory = '/home/ph290/data1/cmip5_data/air_sea_flux/'

files=glob.glob(directory+'fgco2*.nc')
files=np.array(files)

'''
regrid files
'''

# tmp = files[0].split('_')[3]
# tmp2= tmp.split('/')
# variable_name = tmp2[1]

# for file in files:
#     tmp =file.split('/')
#     start = '/'.join(tmp[0:6])
#     end = tmp[6]
#     subprocess.Popen("cdo remapbil,r360x180 -selname,"+variable_name+" "+file+" "+start+"/regridded_"+end,shell=True)
#     #putting on common 1x1 degree grids

#note that this spawns processssed to go off and do this, so takes a while - not the rest of the sxcript will carry on...

'''
get info about files
'''

directory = '/home/ph290/data1/cmip5_data/air_sea_flux/'

files=glob.glob(directory+'regridded*.nc')
files=np.array(files)

model_name=[]
for file in files:
    model_name.append(file.split('_')[6])

model_name = np.array(model_name)
model_name_unique = np.unique(model_name)

ens_member=[]
for file in files:
    tmp = file.split('_')[8]
    ens_member.append(np.int(tmp[1]))

ens_member = np.array(ens_member)

file_year=[]
for file in files:
    tmp = file.split('_')[9]
    file_year.append(np.int(tmp[7:11]))

file_year = np.array(file_year)


'''
read in all cmip5 fgco2 (rcp8.5)
'''

all_cubes=[]
for model in model_name_unique: 
    loc = np.where((model_name == model) & (file_year <= 2100))
    loc2 = np.where(ens_member[loc[0]] == np.min(ens_member[loc[0]]))
    files_tmp=files[loc[0][loc2[0]]]
    cubes = iris.load_raw(files_tmp,callback=my_callback)
    cube = iris.experimental.concatenate.concatenate(cubes)
    cube[0].data = cube[0].data*(60.0*60.0*24.0*30.0)
    if (model == 'BNU-ESM') | (model == 'CanESM2'):
        cube[0].data = cube[0].data*(12/40.0)
    #converting to flux per month for cumulative totals... - NOT WORK OUT ACTUAL NO. DATS IN MONTH FOR EACH MODEL
    all_cubes.append(cube[0])

'''
read in all cmip5 areacello files - note did not use these in the end...
'''

# areacello_directory='/data/data1/ph290/cmip5_data/areacello_files/'

# areacello_files=glob.glob(areacello_directory+'*.nc')
# areacello_files=np.array(areacello_files)

# areacello_model_name=[]
# for file in areacello_files:
#     areacello_model_name.append(file.split('_')[4])

# areacello_model_name = np.array(areacello_model_name)
# areacello_model_name_unique = np.unique(areacello_model_name)

# areacello_cubes=[]
# for model in areacello_model_name_unique:
#     loc=np.where(areacello_model_name == model)
#     tmp_cube=iris.load_cube(areacello_files[loc[0][0]])
#     areacello_cubes.append(tmp_cube)

'''
sum all months together
'''

all_cubes_cumulative=[]
for cube in all_cubes:
    all_cubes_cumulative.append(cube.collapsed(['time'], iris.analysis.SUM))

'''
convert from monthly values to annual values
'''

all_cubes_annual_tot=[]
for cube in all_cubes:
    tmp = monthly_to_yearly(cube)
    tmp.data = tmp.data*12.0
    all_cubes_annual_tot.append(tmp)

'''
and calculate timeseries
'''

all_cubes_timeseries=[]
for cube in all_cubes_annual_tot:
    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds()
    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)
    all_cubes_timeseries.append(cube.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))
    

#array(['BNU-ESM', 'CESM1-BGC', 'CMCC-CESM', 'CanESM2', 'GFDL-ESM2G',
       'GFDL-ESM2M', 'HadGEM2-CC', 'HadGEM2-ES', 'IPSL-CM5A-LR',
       'IPSL-CM5A-MR', 'IPSL-CM5B-LR', 'MIROC-ESM', 'MIROC-ESM-CHEM',
       'MPI-ESM-LR'], 



for i,cube in enumerate(all_cubes_timeseries):
    tmp_data = (all_cubes_timeseries[i].data)
    coord = all_cubes_timeseries[i].coord('time')
    dt = coord.units.num2date(coord.points)
    year = np.array([coord.units.num2date(value).year for value in coord.points])
    plt.plot(year,tmp_data)
    

plt.show()

for cube in all_cubes_cumulative:
    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds()
    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds()

# for i,cube in enumerate(all_cubes_cumulative):
#     qplt.contourf(cube)
#     plt.show()


clean_cubes=[]
for cube in all_cubes_cumulative:
    cube.attributes.clear()
    cube.add_dim_coord
    clean_cubes.append(cube)

clean_cubes2 = iris.experimental.concatenate.concatenate(clean_cubes)

data_array = []
for cube in clean_cubes2:
    data_array.append(cube.data)

data_array=np.array(data_array)

tmp_shape = clean_cubes2[0].data.shape

cmip5_stdev = clean_cubes2[0].copy()
cmip5_stdev.data =cmip5_stdev.data*0.0
for i in np.arange(tmp_shape[0]):
    for j in np.arange(tmp_shape[1]):
        cmip5_stdev.data[i,j]=np.std(data_array[:,i,j])

cmip5_stdev.data.mask = clean_cubes2[0].data.mask

cmip5_mean = clean_cubes2[0].copy()
cmip5_mean.data =cmip5_mean.data*0.0
for i in np.arange(tmp_shape[0]):
    for j in np.arange(tmp_shape[1]):
        cmip5_mean.data[i,j]=np.mean(data_array[:,i,j])

cmip5_mean.data.mask = clean_cubes2[0].data.mask


plt.figure()
qplt.contourf(cmip5_stdev,np.linspace(0,5,50))
plt.title('Standard deviation in CMIP5 cumulative CO$_2$ flux by 2005-2100 (RCP8.5)')
plt.gca().coastlines()
plt.savefig('/home/ph290/Desktop/delete/cum_as_flux_sd.pdf')
#plt.show()                      

plt.figure()
qplt.contourf(cmip5_mean,np.linspace(-5,10,50))
plt.title('mean CMIP5 cumulative CO$_2$ flux by 2005-2100 (RCP8.5)')
plt.gca().coastlines()
plt.savefig('/home/ph290/Desktop/delete/cum_as_flux.pdf')
#plt.show()
