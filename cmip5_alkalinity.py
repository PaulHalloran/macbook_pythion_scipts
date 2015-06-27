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

def monthly_to_yearly(cube):
    try:
        iris.coord_categorisation.add_year(cube, 'time', name='year2')
    except ValueError:
        cube_tmp = cube.aggregated_by('year2', iris.analysis.MEAN)
        return cube_tmp

def regrid_data_0(file,variable_name,out_filename):
    p = subprocess.Popen("cdo remapbil,r360x180 -selname,"+variable_name+" "+file+" "+out_filename,shell=True)
    p.wait()

def my_callback(cube,field, files):
    # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
    cube.attributes.pop('history')
    cube.attributes.pop('tracking_id')
    cube.attributes.pop('creation_date')
    cube.attributes.pop('table_id')
    #if np.size(cube) > 1:
    #cube = iris.experimental.concatenate.concatenate(cube)
    return cube

def extract_data(data_in):
    data_out = data_in.data
    return data_out

directory = '/data/data0/ph290/cmip5_data/'

alk_files = np.array(glob.glob(directory+'talk/*.nc'))
so_files = np.array(glob.glob(directory+'so/*.nc'))

'''
which models do we have?
'''

models = []
for file in alk_files:
    models.append(file.split('/')[-1].split('_')[2])

alk_models_unique = np.unique(np.array(models))

models = []
for file in so_files:
    models.append(file.split('/')[-1].split('_')[2])

so_models_unique = np.unique(np.array(models))


models_unique = np.intersect1d(so_models_unique,alk_models_unique)
#list of only thos emodels that are in both lists


'''
reads in and regrid data into a single 360x180 file per model - note, this only needs to be done once
'''

tmp_dir = '/data/data0/ph290/cmip5_data/tmp_dir/'

# final_models1 = []
# cubes = []
# for i,model in enumerate(models_unique):
#     print i
#     file = glob.glob(directory+'talk/'+'*'+model+'*.nc')
#     cube = iris.load(file,'sea_water_alkalinity_expressed_as_mole_equivalent')
#     length = len(cube)
#     cube_tmp = cube[0]
#     cube_data = cube_tmp[0].data*0.0
#     count = 0
#     #note that the following is a very messy way to do things, but the only way that I did not run into memory problems...
#     for j in np.arange(length):
#         cube = iris.load(file,'sea_water_alkalinity_expressed_as_mole_equivalent')
#         cube_tmp = extract_data(cube[j])
#         shape = np.shape(cube_tmp)
#         cube_data = np.sum([cube_data,np.sum(cube_tmp,axis = 0)],axis = 0)
#         count += shape[0]
#         print count
#         del cube_tmp
#         del cube
#         dummp = gc.collect()
#     cube = iris.load(file,'sea_water_alkalinity_expressed_as_mole_equivalent')
#     cube_tmp = cube[0]
#     div_array = (cube_tmp[0].data*0.0)+count
#     data = np.divide(cube_data,div_array)
#     cube = cube_tmp[0]
#     cube.data = data
#     iris.fileformats.netcdf.save(cube, tmp_dir+'delete.nc', netcdf_format='NETCDF3_CLASSIC')
#     out_filename = directory+'talk/regridded/'+model+'.nc'
#     regrid_data_0(tmp_dir+'delete.nc','talk',out_filename)
#     #note that the regridding step messus up the depth coordinate...
#     final_models1.append(model)
#     print model + ' done'

# '''
# and salinity
# '''

# final_models2 = []
# for i,model in enumerate(models_unique):
#     print i
#     file = glob.glob(directory+'so/'+'*'+model+'*.nc')
#     cube = iris.load(file,'sea_water_salinity')
#     length = len(cube)
#     cube_tmp = cube[0]
#     cube_data = cube_tmp[0].data*0.0
#     count = 0
#     for j in np.arange(length):
#         cube = iris.load(file,'sea_water_salinity')
#         cube_tmp = extract_data(cube[j])
#         shape = np.shape(cube_tmp)
#         cube_data = np.sum([cube_data,np.sum(cube_tmp,axis = 0)],axis = 0)
#         count += shape[0]
#         print count
#         del cube_tmp
#         del cube
#         dummp = gc.collect()
#     cube = iris.load(file,'sea_water_salinity')
#     cube_tmp = cube[0]
#     div_array = (cube_tmp[0].data*0.0)+count
#     data = np.divide(cube_data,div_array)
#     cube = cube_tmp[0]
#     cube.data = data
#     iris.fileformats.netcdf.save(cube, tmp_dir+'delete.nc', netcdf_format='NETCDF3_CLASSIC')
#     out_filename = directory+'so/regridded/'+model+'.nc'
#     regrid_data_0(tmp_dir+'delete.nc','so',out_filename)
#     final_models2.append(model)
#     print model + ' done'
# 
# final_models = np.intersect1d(final_models2,final_models1)

'''
Read in 1x1 degree basin masks from WOA
'''

woa_dir = '/home/ph290/Documents/teaching/'
basin_mask = iris.load_cube(woa_dir+'basin.nc')
basin_mask_tmp = basin_mask[0][0]
basin_mask_tmp.data[np.where(np.logical_not(basin_mask_tmp.data == 1))] = np.nan
# 1 = Atlantic
# 2 = Pacific
# 3 = Indian
# 4 = Southen Ocean
# 5 = Arctic
# if upside down:
# basin_mask_flipped = iris.analysis.maths.np.flipud(basin_mask.data)
# apply mask with:
# sstb = iris.analysis.maths.multiply(sst,basin_mask_flipped)

'''
Read in GLODAP and WOA data
'''

qlodap_dir = '/home/ph290/data1/observations/glodap/'
woa_dir = '/home/ph290/Documents/teaching/'
alk_in = iris.load_cube(qlodap_dir+'PALK.nc','Potential_Alkalinity')
alk_in = iris.analysis.maths.divide(alk_in,1000.0) # convert units
alk_in.transpose((0,2,1)) #reorders dimenions to be same as CMIP5 and WOA
salinity_in = iris.load_cube(woa_dir+'salinity_annual_1deg.nc','sea_water_salinity')



'''
Reading file details (model name etc)
'''

alk_files_regrid = np.array(glob.glob(directory+'talk/regridded/*.nc'))

model_names = []
for file in alk_files_regrid:
    tmp = file.split('/')[-1].split('_')[0].split('.')[0]
    model_names.append(tmp)
  
'''
Reading in regridded files
'''

model_names = np.array(model_names)
model_names_unique = np.unique(model_names)

alk_cubes = []
for model in model_names_unique:
    print model
    files = np.array(glob.glob(directory+'talk/regridded/*'+model+'*.nc'))
    cube = iris.load(files)
    alk_cubes.append(cube[0])

so_cubes = []
for model in model_names_unique:
    print model
    files = np.array(glob.glob(directory+'so/regridded/*'+model+'*.nc'))
    cube = iris.load(files)
    cube = cube[0]
    mean_tmp = cube.collapsed(['latitude','longitude'],iris.analysis.MEAN)
    if mean_tmp[0].data <= 1:
        cube = iris.analysis.maths.multiply(cube,1000.0)
        print 'converting model because in incorrect units'
    so_cubes.append(cube)

basins = np.array([0,1,2,3,10,11])
basin_names = np.array(['Global','Atlantic','Pacific','Indian','Southern Ocean','Arctic Ocean'])

'''
Depths
'''

depths = []
for model in model_names_unique:
    print model
    count = 0
    files = glob.glob(directory+'talk/'+'*'+model+'*.nc')
    cube = iris.load(files[0],'sea_water_alkalinity_expressed_as_mole_equivalent')
    try:
        depth = cube[0].coord('ocean depth coordinate').points
    except iris.exceptions.CoordinateNotFoundError:
        print 'depth name not ocean depth coordinate'
        count += 1
    try:
        depth = cube[0].coord('depth').points
    except iris.exceptions.CoordinateNotFoundError:
        print 'depth name not depth'
        count += 1
    try:
        depth = cube[0].coord('ocean sigma over z coordinate').points
    except iris.exceptions.CoordinateNotFoundError:
        print 'depth name not sigma'
        count += 1
    if count == 3:
        print 'no depth coordinate was found'
        break
    depths.append(depth)



'''
Calculate CMIP5 profiles
'''

regional_profiles_alk = []
regions = ['']

for i in basins:
    model_profiles = []
    for cube in alk_cubes:
        if not cube.coord('latitude').has_bounds():
            cube.coord('latitude').guess_bounds()
        if not cube.coord('longitude').has_bounds():
            cube.coord('longitude').guess_bounds()
        grid_areas = iris.analysis.cartography.area_weights(cube)
        basin_mask_tmp = basin_mask[0][0]
        if i == 0:
            #global case
            loc1 = np.where(np.logical_not(basin_mask_tmp.data >= 1))
            loc2 = np.where(basin_mask_tmp.data >= 1)
        else:
            #all other cases
            loc1 = np.where(np.logical_not(basin_mask_tmp.data == i))
            loc2 = np.where(basin_mask_tmp.data == i)
        basin_mask_tmp.data[loc1] = 0.0
        basin_mask_tmp.data[loc2] = 1.0
        basin_mask_flipped = basin_mask_tmp
        basin_mask_flipped.data = iris.analysis.maths.np.flipud(basin_mask_tmp.data)
        cube_tmp = cube
        cube_tmp = iris.analysis.maths.multiply(cube,basin_mask_flipped)
        cube_tmp.data = ma.masked_equal(cube_tmp.data,0)
        data_shape = np.shape(cube_tmp.data)
        a = ma.getmaskarray(alk_in[0].data)
        a =np.roll(a,180,axis=1)
        a2 = np.tile(a,(data_shape[0],1,1))
        b = ma.getmaskarray(cube_tmp.data)
        mask = a2 | b
        #by combining the masks above, the CMIP5 data has exactky the same land sea mask as the WOA data
        cube_tmp.data = ma.masked_array(cube_tmp.data,mask)
        #qplt.contourf(cube_tmp[0])
        #plt.gca().coastlines()
        #plt.show()
        model_profiles.append(cube_tmp.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))
    regional_profiles_alk.append(model_profiles)


'''
and so
'''

regional_profiles_so = []
regions = ['']

for i in basins:
    model_profiles = []
    for cube in so_cubes:
        if not cube.coord('latitude').has_bounds():
            cube.coord('latitude').guess_bounds()
        if not cube.coord('longitude').has_bounds():
            cube.coord('longitude').guess_bounds()
        grid_areas = iris.analysis.cartography.area_weights(cube)
        basin_mask_tmp = basin_mask[0][0]
        if i == 0:
            #global case
            loc1 = np.where(np.logical_not(basin_mask_tmp.data >= 1))
            loc2 = np.where(basin_mask_tmp.data >= 1)
        else:
            #all other cases
            loc1 = np.where(np.logical_not(basin_mask_tmp.data == i))
            loc2 = np.where(basin_mask_tmp.data == i)
        basin_mask_tmp.data[loc1] = 0.0
        basin_mask_tmp.data[loc2] = 1.0
        basin_mask_flipped = basin_mask_tmp
        basin_mask_flipped.data = iris.analysis.maths.np.flipud(basin_mask_tmp.data)
        cube_tmp = cube
        cube_tmp = iris.analysis.maths.multiply(cube,basin_mask_flipped)
        cube_tmp.data = ma.masked_equal(cube_tmp.data,0)
        data_shape = np.shape(cube_tmp.data)
        a = ma.getmaskarray(alk_in[0].data)
        a =np.roll(a,180,axis=1)
        a2 = np.tile(a,(data_shape[0],1,1))
        b = ma.getmaskarray(cube_tmp.data)
        mask = a2 | b
        #by combining the masks above, the CMIP5 data has exactky the same land sea mask as the WOA data
        cube_tmp.data = ma.masked_array(cube_tmp.data,mask)
        #qplt.contourf(cube_tmp[0])
        #plt.gca().coastlines()
        #plt.show()
        model_profiles.append(cube_tmp.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))
    regional_profiles_so.append(model_profiles)


'''
and alk/so (normalised)
'''

cubes = np.copy(alk_cubes)
for i in np.arange(len(cubes)):
    cubes[i] = iris.analysis.maths.divide(alk_cubes[i],so_cubes[i])

regional_profiles_normal = []
regions = ['']

for i in basins:
    model_profiles = []
    for cube in cubes:
        if not cube.coord('latitude').has_bounds():
            cube.coord('latitude').guess_bounds()
        if not cube.coord('longitude').has_bounds():
            cube.coord('longitude').guess_bounds()
        grid_areas = iris.analysis.cartography.area_weights(cube)
        basin_mask_tmp = basin_mask[0][0]
        if i == 0:
            #global case
            loc1 = np.where(np.logical_not(basin_mask_tmp.data >= 1))
            loc2 = np.where(basin_mask_tmp.data >= 1)
        else:
            #all other cases
            loc1 = np.where(np.logical_not(basin_mask_tmp.data == i))
            loc2 = np.where(basin_mask_tmp.data == i)
        basin_mask_tmp.data[loc1] = 0.0
        basin_mask_tmp.data[loc2] = 1.0
        basin_mask_flipped = basin_mask_tmp
        basin_mask_flipped.data = iris.analysis.maths.np.flipud(basin_mask_tmp.data)
        cube_tmp = cube
        cube_tmp = iris.analysis.maths.multiply(cube,basin_mask_flipped)
        cube_tmp.data = ma.masked_equal(cube_tmp.data,0)
        data_shape = np.shape(cube_tmp.data)
        a = ma.getmaskarray(alk_in[0].data)
        a =np.roll(a,180,axis=1)
        a2 = np.tile(a,(data_shape[0],1,1))
        b = ma.getmaskarray(cube_tmp.data)
        mask = a2 | b
        #by combining the masks above, the CMIP5 data has exactky the same land sea mask as the WOA data
        cube_tmp.data = ma.masked_array(cube_tmp.data,mask)
        #qplt.contourf(cube_tmp[0])
        #plt.gca().coastlines()
        #plt.show()
        model_profiles.append(cube_tmp.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))
    regional_profiles_normal.append(model_profiles)


'''
Calculate GLODAP/WOA profiles
'''

#alk_in = iris.load_cube(qlodap_dir+'PALK.nc','Potential_Alkalinity')
#salinity_in = iris.load_cube(woa_dir+'salinity_annual_1deg.nc','sea_water_salinity')

so_in = salinity_in[0]
so_in_data = np.roll(so_in.data,180,axis = 2)
cube_tmp = alk_in.copy()
cube_tmp.data = so_in_data
cube = iris.analysis.maths.divide(alk_in,cube_tmp)

obs_profiles = []
for i in basins:
    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds()
    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)
    basin_mask_tmp = basin_mask[0][0]
    if i == 0:
        #global case
        loc1 = np.where(np.logical_not(basin_mask_tmp.data >= 1))
        loc2 = np.where(basin_mask_tmp.data >= 1)
    else:
        #all other cases
        loc1 = np.where(np.logical_not(basin_mask_tmp.data == i))
        loc2 = np.where(basin_mask_tmp.data == i)
    basin_mask_tmp.data[loc1] = 0.0
    basin_mask_tmp.data[loc2] = 1.0
    basin_mask_flipped = basin_mask_tmp
    basin_mask_flipped.data = iris.analysis.maths.np.flipud(basin_mask_tmp.data)
    basin_mask_flipped.data = iris.analysis.maths.np.roll(basin_mask_tmp.data,180,axis=1)
    cube_tmp = cube
    cube_tmp = iris.analysis.maths.multiply(cube,basin_mask_flipped)
    cube_tmp.data = ma.masked_equal(cube_tmp.data,0)
    #qplt.contourf(cube_tmp[0])
    #plt.gca().coastlines()
    #plt.show()
    obs_profiles.append(cube_tmp.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))

'''
obs profile alk
'''

cube = alk_in

obs_profiles_alk = []
for i in basins:
    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds()
    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)
    basin_mask_tmp = basin_mask[0][0]
    if i == 0:
        #global case
        loc1 = np.where(np.logical_not(basin_mask_tmp.data >= 1))
        loc2 = np.where(basin_mask_tmp.data >= 1)
    else:
        #all other cases
        loc1 = np.where(np.logical_not(basin_mask_tmp.data == i))
        loc2 = np.where(basin_mask_tmp.data == i)
    basin_mask_tmp.data[loc1] = 0.0
    basin_mask_tmp.data[loc2] = 1.0
    basin_mask_flipped = basin_mask_tmp
    basin_mask_flipped.data = iris.analysis.maths.np.flipud(basin_mask_tmp.data)
    basin_mask_flipped.data = iris.analysis.maths.np.roll(basin_mask_tmp.data,180,axis=1)
    cube_tmp = cube
    cube_tmp = iris.analysis.maths.multiply(cube,basin_mask_flipped)
    cube_tmp.data = ma.masked_equal(cube_tmp.data,0)
    #qplt.contourf(cube_tmp[0])
    #plt.gca().coastlines()
    #plt.show()
    obs_profiles_alk.append(cube_tmp.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))


'''
Plot
'''

linestyles = ['-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':']

for i,profiles in enumerate(regional_profiles_normal):
    plt.figure()
    print i
    for j,model_profile in enumerate(profiles):
        line = plt.plot(model_profile.data,depths[j]*(-1.0))
        plt.setp(line, linestyle=linestyles[j],linewidth = 2)
        #plt.title(basin_names[i])
    #qplt.plot(obs_profiles[i],'k',linewidth = 3)
    #print 'obs next'
    #print obs_profiles[i].data
    plt.plot(obs_profiles[i].data,obs_profiles[i].coord('depth').points*(-1.0),'k',linewidth = 3)
    plt.title(basin_names[i])
    #plt.show()
    #plt.savefig('/home/ph290/Desktop/delete/'+basin_names[i]+'.png')

for i,profiles in enumerate(regional_profiles_alk):
    plt.figure()
    print i
    for j,model_profile in enumerate(profiles):
        line = plt.plot(model_profile.data,depths[j]*(-1.0))
        plt.setp(line, linestyle=linestyles[j],linewidth = 2)
        #plt.title(basin_names[i])
    #qplt.plot(obs_profiles[i],'k',linewidth = 3)
    #print 'obs next'
    #print obs_profiles[i].data
    plt.plot(obs_profiles_alk[i].data,obs_profiles_alk[i].coord('depth').points*(-1.0),'k',linewidth = 3)
    plt.xlim([2.3,2.6])
    plt.title(basin_names[i])
    #plt.show()
    plt.savefig('/home/ph290/Desktop/delete/'+basin_names[i]+'.pdf')

for j,model_profile in enumerate(profiles):
    line = plt.plot([0,1],[j+1,j+1])
    plt.text(1.2, j+0.8, model_names_unique[j],fontsize=12)
    plt.setp(line, linestyle=linestyles[j],linewidth = 2)
    plt.xlim([0,2])
    plt.ylim([0,18])

#plt.show()
plt.savefig('/home/ph290/Desktop/delete/lege.png')

#now need to  do the salinity side of things...

