'''
This script produces composites of sst from high and low AMOC periods of CMIP5 conbtrol runs.
The control runs are filtered to move the interannual and the multicentenial variability to remove noise and drift. The signal should therefore essentially be the AMO-like variability as expressed in the ssts
'''


import numpy as np
import numpy.ma as ma
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
import matplotlib as mpl
import gc
import scipy
import scipy.signal as signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
import monthly_to_yearly as m2yr
import matplotlib.mlab as ml
import cartopy
import cartopy.feature as cfeature

def regridding_unstructured(cube):
    lats = cube.coord('latitude').points
    lons = cube.coord('longitude').points
    tmp_shape_lats = lats.shape
    tmp_shape_lons = lons.shape
    if len(tmp_shape_lats) == 1:
        cube2 = iris.cube.Cube(np.zeros((180, 360), np.float32),standard_name='sea_surface_temperature', long_name='Sea Surface Temperature', var_name='tos', units='K',dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
        out =  iris.analysis.interpolate.regrid(cube, cube2)
        return out.data,out.data,out.data
    else:
        data = np.array(cube.data)
        lats2 = np.reshape(lats,lats.shape[0]*lats.shape[1])
        lons2 = np.reshape(lons,lons.shape[0]*lons.shape[1])
        data2 = np.reshape(data,data.shape[0]*data.shape[1])
        yi = np.linspace(-90.0,90.0,180.0)
        xi = np.linspace(0.0,360.0,360.0)
        zi = ml.griddata(lons2,lats2,data2,xi,yi)
        return xi,yi,zi



def my_callback(cube,field, files):
    # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
    cube.attributes.pop('tracking_id')
    cube.attributes.pop('creation_date')
    cube.attributes.pop('table_id')
    #if np.size(cube) > 1:
    #cube = iris.experimental.concatenate.concatenate(cube)


def extract_data(data_in):
    data_out = data_in.data
    return data_out


def regrid_data_0(file,variable_name,out_filename):
    p = subprocess.Popen("cdo remapbil,r360x180 -selname,"+variable_name+" "+file+" "+out_filename,shell=True)
    p.wait()
    return iris.load(out_filename)

directory = '/home/ph290/data0/cmip5_data/'
files1 = np.array(glob.glob(directory+'msftmyz/piControl/*.nc'))
files2 = np.array(glob.glob(directory+'tos/piControl/*.nc'))

'''
which models do we have?
'''

models1 = []
for file in files1:
    models1.append(file.split('/')[-1].split('_')[2])


models_unique1 = np.unique(np.array(models1))

models2 = []
for file in files2:
    models2.append(file.split('/')[-1].split('_')[2])


models_unique2 = np.unique(np.array(models2))

models_unique = np.intersect1d(models_unique1,models_unique2)


'''
read in AMOC
'''

def remove_anonymous(cube, field, filename):
# this only loads in the region relating to the atlantic
	cube.attributes.pop('creation_date')
	cube.attributes.pop('tracking_id')
	cube = cube[:,0,:,:]
	return cube

cmip5_max_strmfun = []
cmip5_year = []

for model in models_unique:
    print model
    files = np.array(glob.glob(directory+'msftmyz/piControl/*'+model+'*.nc'))
    cubes = iris.load(files,'ocean_meridional_overturning_mass_streamfunction',callback = remove_anonymous)
    max_strmfun = []
    year = []
    month = []
    for cube_tmp in cubes:
        cube = cube_tmp.copy()
        cube = m2yr.monthly_to_yearly(cube)
        for i,sub_cube in enumerate(cube.slices(['latitude', 'depth'])):
            max_strmfun.append(np.max(sub_cube.data))
            coord = sub_cube.coord('time')
            year.append(np.array([coord.units.num2date(value).year for value in coord.points])[0])
    cmip5_max_strmfun.append(np.array(max_strmfun))
    cmip5_year.append(np.array(year))
    


cmip5_max_strmfun = np.array(cmip5_max_strmfun)
cmip5_year = np.array(cmip5_year)

b, a = scipy.signal.butter(3, 0.01, 'high')
#here we are generating a high-pass filter to allow us to remove all the really low-period stuff - 200 year typw variability - this will hopefully mean that we're not picking up any signal from drift etc.
#an argument coudl be made to also filter out all of the really high frequenfy stuff - have a play with this...

high_years = []
low_years = []
for i,years in enumerate(cmip5_year):
    sorting = np.argsort(np.array(years))
    ts = cmip5_max_strmfun[i][sorting]-np.mean(cmip5_max_strmfun[i][sorting])
    b, a = scipy.signal.butter(3, 0.01, 'high')
    #here we are generating a high-pass filter to allow us to remove all the really low-period stuff - 200 year typw variability - this will hopefully mean that we're not picking up any signal from drift etc.
    low_pass_timeseries = scipy.signal.filtfilt(b, a, ts)
    b, a = scipy.signal.butter(3, 0.1, 'low')
    #And then getting rid of the very high frequency stuff
    low_pass_timeseriesb = scipy.signal.filtfilt(b, a, low_pass_timeseries)
    high_years.append(np.unique(years[np.where(low_pass_timeseriesb > 0.0)]))
    low_years.append(np.unique(years[np.where(low_pass_timeseriesb < 0.0)]))

'''
examining filteroing etc - kind of for dubuggingish stuff, so commented out
'''

# i = 0
# years = cmip5_year[0]
# sorting = np.argsort(np.array(years))
# strfun = np.array(cmip5_max_strmfun[i])
# ts = strfun[sorting]-np.mean(strfun[sorting])
# b, a = scipy.signal.butter(3, 0.01, 'high')
# low_pass_timeseries = scipy.signal.filtfilt(b, a, ts)
# b, a = scipy.signal.butter(3, 0.01, 'low')
# low_pass_timeseries2 = scipy.signal.filtfilt(b, a, ts)
# b, a = scipy.signal.butter(3, 0.1, 'low')
# low_pass_timeseries3 = scipy.signal.filtfilt(b, a, low_pass_timeseries)
# plt.plot(ts)
# plt.plot(low_pass_timeseries)
# plt.plot(low_pass_timeseries2)
# plt.plot(low_pass_timeseries3,linewidth = 3)
# plt.show()

# powers, freqs = mlab.psd(ts)
# powers2, freqs2 = mlab.psd(low_pass_timeseries)
# plt.plot(freqs*len(ts),powers)
# plt.plot(freqs2*len(low_pass_timeseries),powers2)
# plt.xlim(0,200)
# plt.ylim(0,10e18)
# plt.show()


cmip5_cube_high_amo_mean = []
cmip5_cube_low_amo_mean = []

for k,model in enumerate(models_unique):
    files = np.array(glob.glob(directory+'tos/piControl/*'+model+'*.nc'))
    cubes = iris.load(files,'sea_surface_temperature',callback = my_callback)
    cube_tmp = cubes[0][0].copy()
    high_amo_data = cube_tmp.data*0.0
    low_amo_data = cube_tmp.data*0.0
    high_amo_count = 0.0
    low_amo_count = 0.0
    for j,cube in enumerate(cubes):
        print np.str(j)+' of '+np.str(len(cubes))
        cube = m2yr.monthly_to_yearly(cube)
        length = cube.shape[0]
        dim1_name = cube.coords()[0].standard_name
        dim2_name = cube.coords()[1].standard_name
        if dim1_name == None:
            dim1_name = cube.coords()[0].long_name
            dim2_name = cube.coords()[1].long_name
        for i,sub_cube in enumerate(cube.slices([dim1_name,dim2_name])):
        #for i,sub_cube in enumerate(cube.slices(['cell index along second dimension', 'cell index along first dimension'])):
            print  np.str(i)+' of '+np.str(length)
            tmp_cube = sub_cube.copy()
            #copy for memory reasons
            coord = tmp_cube.coord('time')
            yr_tmp = (np.array([coord.units.num2date(value).year for value in coord.points]))[0]
            if np.in1d(np.array(yr_tmp), np.array(high_years[k]))[0]:
                high_amo_data = np.sum([high_amo_data,tmp_cube.data],axis = 0, out=high_amo_data)
                high_amo_count += 1.0
            if np.in1d(np.array(yr_tmp), np.array(low_years[k]))[0]:
                #low_amo_data += tmp_cube.data
                low_amo_data = np.sum([low_amo_data,tmp_cube.data],axis = 0, out=low_amo_data)
                low_amo_count += 1.0
    div_array = (cubes[0][0].copy().data*0.0)+high_amo_count
    div_cube = cube_tmp.copy()
    div_cube.data = div_array
    high_amo_cube = cube_tmp.copy()
    high_amo_cube.data = high_amo_data.copy()
    cube_high_amo_mean = cube_tmp.copy()
    cube_high_amo_mean = iris.analysis.maths.divide(high_amo_cube,div_cube)
    div_array = (cubes[0][0].copy().data*0.0)+low_amo_count
    div_cube = cube_tmp.copy()
    div_cube.data = div_array
    low_amo_cube = cube_tmp.copy()
    low_amo_cube.data = low_amo_data.copy()
    cube_low_amo_mean = cube_tmp.copy()
    cube_low_amo_mean = iris.analysis.maths.divide(low_amo_cube,div_cube)
    cmip5_cube_high_amo_mean.append(cube_high_amo_mean)
    cmip5_cube_low_amo_mean.append(cube_low_amo_mean)

'''
High AMOC minus low AMOC ssts:
'''

cmip5_sst_amoc_diff = []

for k,model in enumerate(models_unique):
    cube = iris.analysis.maths.subtract(cmip5_cube_high_amo_mean[k],cmip5_cube_low_amo_mean[k])
    cmip5_sst_amoc_diff.append(cube)

'''
Regrid outputs to 1x1 degree for comparision
Note that this fails - if I write these fields out using  iris.fileformats.netcdf.save they are not in a suitable format for regridding with cdo remapbil
'''

# tmp_dir = '/data/data0/ph290/cmip5_data/tmp_dir/'

# cmip5_sst_amoc_diff_regridded = []

# for cube in cmip5_sst_amoc_diff:
#     iris.fileformats.netcdf.save(cube, tmp_dir+'delete.nc', netcdf_format='NETCDF3_CLASSIC')
#     out_filename = directory+'amoc_composits/'+model+'_tos.nc'
#     tmp = regrid_data_0(tmp_dir+'delete.nc','sea_surface_temperature',out_filename)
#     cmip5_sst_amoc_diff_regridded.append(tmp)

# for k,model in enumerate(models_unique):
#this does not seem to be working so well...
#     tmp = cmip5_sst_amoc_diff[k]
#     xi,yi,zi = regridding_unstructured(tmp)
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     zi2 = np.roll(zi,110,axis = 1)
#     ax.contourf(xi,yi,zi2,50,transform=ccrs.PlateCarree())
#     ax.coastlines()
#     #ax.add_feature(cartopy.feature.LAND)
#     #ax.set_global()
#     plt.show()
#     #plt.savefig('/home/ph290/Desktop/delete/'+model+'_tos_amoc.png')


# for k,model in enumerate(models_unique):
#     plt.figure()
#     tmp = cmip5_sst_amoc_diff[k]
#     try:
#         lat_tmp = tmp.coord('latitude').points[0][0]
#         lon_tmp = tmp.coord('longitude').points[0][0]
#     except IndexError:
#         lat_tmp = tmp.coord('latitude').points[0]
#         lon_tmp = tmp.coord('longitude').points[0]
#     print lat_tmp,lon_tmp
#     if lat_tmp >= 0:
#         tmp.data = np.flipud(tmp.data)
#     if lon_tmp >= 190:
#         tmp.data = np.roll(tmp.data,180,axis = 1)
#     plt.contourf(tmp.data, (np.arange(100)/100.0)-0.5)
#     plt.title(model)
#     plt.savefig('/home/ph290/Desktop/delete/'+model+'_tos_amoc.png')
#     #plt.show()

'''
BETTER GRIDS...
'''

seems to fail on the 6th model

  Natgrid - two input triples have the same x/y coordinates
            but different data values: 

                First triple:  79.816010 300.085327 0.000305
                Second triple: 79.816010 300.085327 -0.002716




for k,model in enumerate(models_unique):
    print k
    tmp = cmip5_sst_amoc_diff[k].copy()
    latitude = DimCoord(np.arange(-90,90, 1), standard_name='latitude',units='degrees')
    longitude = DimCoord(np.arange(0, 360, 1), standard_name='longitude',
                         units='degrees')
    cube = iris.cube.Cube(np.zeros((180, 360), np.float32),standard_name='sea_surface_temperature', long_name='Sea Surface Temperature', var_name='tos', units='K',dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
    cube_tmp = regridding_unstructured(tmp)
    x = np.ma.array(cube_tmp[2])
    x = np.ma.masked_where(x == 0,x)
    cube.data = np.roll(x.copy(),360-40,axis = 1)
    #this is arbitrary, and does not solve the issue...
    plt.figure()
    qplt.contourf(cube,50)
    plt.title(model+' AMOC SST composite')
    #plt.gca().coastlines()
    #plt.gca().add_feature(cfeature.LAND)
    plt.savefig('/home/ph290/Desktop/delete/'+model+'_tos_amoC_correct_question.png')
    #plt.show()



# '''
# And plot the data
# '''


# sst = np.subtract(cube_high_amo_mean.data,cube_low_amo_mean.data)
# plt.contourf(np.flipud(sst), (np.arange(100)/200.0)-0.25)
# plt.show()

# pr = iris.analysis.maths.np.subtract(cube_high_amo_mean_pr,cube_low_amo_mean_pr)
# qplt.contourf(pr,60)
# plt.gca().coastlines()
# plt.show()
