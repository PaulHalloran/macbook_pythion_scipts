# import numpy as np
# import numpy.ma as ma
# import matplotlib.pyplot as plt
# import iris
# import glob
# import iris.experimental.concatenate
# import iris.analysis
# import iris.quickplot as qplt
# import iris.analysis.cartography
# import cartopy.crs as ccrs
# import subprocess
# from iris.coords import DimCoord
# import iris.coord_categorisation
# import matplotlib as mpl
# import gc
# import scipy
# import scipy.signal as signal
# from scipy.signal import kaiserord, lfilter, firwin, freqz
# import monthly_to_yearly as m2yr
# from matplotlib import mlab
# import matplotlib.mlab as ml
# import cartopy
# import monthly_to_yearly
# import scipy
# from scipy import signal
# from scipy.signal import butter, lfilter
# import cartopy.feature as cfeature
# from matplotlib.colors import BoundaryNorm



# def cube_extract_region(cube,min_lat,min_lon,max_lat,max_lon):
#     region = iris.Constraint(longitude=lambda v: min_lon <= v <= max_lon,latitude=lambda v: min_lat <= v <= max_lat)
#     return cube.extract(region)



# def my_callback(cube,field, files):
#     # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
#     cube.attributes.pop('tracking_id')
#     cube.attributes.pop('creation_date')
#     cube.attributes.pop('table_id')
# #     cube.attributes.pop('mo_runid')
#     cube.attributes.pop('history')
#     #if np.size(cube) > 1:
#     #cube = iris.experimental.concatenate.concatenate(cube)

# def load_files(in_file,areacello_file):
# 	cube = iris.load_cube(in_file,callback = my_callback)
# 	cube = monthly_to_yearly.monthly_to_yearly(cube)
# 	areacello = iris.load_cube(areacello_file)
# 	mean = cube.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights = np.tile(areacello.data,[cube.shape[0],1,1]))
# 	return mean, cube

# cmap = plt.get_cmap('gist_earth')
# cmap2 = plt.get_cmap('BrBG')

# '''
# dms flux
# '''

# directory = '/home/ph290/data0/cmip5_data/dms/'
# files1 = np.array(glob.glob(directory+'hadgem2es/rcp85/*.nc'))
# files2 = np.array(glob.glob(directory+'hadgem2es/rcp45/*.nc'))
# files3 = np.array(glob.glob(directory+'hadgem2es/rcp26/*.nc'))
# files1b = np.array(glob.glob(directory+'mpiesmlr/rcp85/*.nc'))
# files2b = np.array(glob.glob(directory+'mpiesmlr/rcp45/*.nc'))
# files3b = np.array(glob.glob(directory+'mpiesmlr/rcp26/*.nc'))

# hadgem_areacello_file = '/home/ph290/data1/cmip5_data/areacello_files/areacello_fx_HadGEM2-ES_piControl_r0i0p0.nc'
# mpi_areacello_file = '/home/ph290/data1/cmip5_data/areacello_files/areacello_fx_MPI-ESM-LR_piControl_r0i0p0.nc'

# hadgem_85_mean, hadgem_85 = load_files(files1,hadgem_areacello_file)
# hadgem_45_mean, hadgem_45 = load_files(files2,hadgem_areacello_file)
# hadgem_26_mean, hadgem_26 = load_files(files3,hadgem_areacello_file)

# mpi_85_mean, mpi_85 = load_files(files1b,mpi_areacello_file)
# # mpi_45_mean, mpi_45 = load_files(files2b,mpi_areacello_file)
# mpi_26_mean, mpi_26 = load_files(files3b,mpi_areacello_file)

# '''
# DMS concentration
# '''

# directory_2 = '/home/ph290/data0/cmip5_data/dms_conc/'
# files0_2 = np.array(glob.glob(directory_2+'hadgem2es/rcp26/*.nc'))
# files1_2 = np.array(glob.glob(directory_2+'hadgem2es/rcp85/*.nc'))
# files2_2 = np.array(glob.glob(directory_2+'hadgem2es/hist/*.nc'))
# files0b_2 = np.array(glob.glob(directory_2+'mpiesmlr/rcp26/*.nc'))
# files1b_2 = np.array(glob.glob(directory_2+'mpiesmlr/rcp85/*.nc'))
# files2b_2 = np.array(glob.glob(directory_2+'mpiesmlr/hist/*.nc'))

# hadgem_26_mean_conc, hadgem_26_conc = load_files(files0_2,hadgem_areacello_file)
# hadgem_85_mean_conc, hadgem_85_conc = load_files(files1_2,hadgem_areacello_file)
# hadgem_hist_mean_conc, hadgem_hist_conc = load_files(files2_2,hadgem_areacello_file)
# hadgem_1985_2005_avg = hadgem_hist_conc[-19:-1].copy().collapsed('time',iris.analysis.MEAN)

# mpi_26_mean_conc, mpi_26_conc = load_files(files0b_2,mpi_areacello_file)
# mpi_85_mean_conc, mpi_85_conc = load_files(files1b_2,mpi_areacello_file)
# mpi_hist_mean_conc, mpi_hist_conc = load_files(files2b_2,mpi_areacello_file)
# mpi_1985_2005_avg = mpi_hist_conc[-19:-1].copy().collapsed('time',iris.analysis.MEAN)


# '''
# execfile('dms_cmip5.py')
# '''

# plt.figure()
# qplt.plot(hadgem_85_mean,'r')
# # qplt.plot(hadgem_45_mean,'g')
# qplt.plot(hadgem_26_mean,'b')
# qplt.plot(iris.analysis.maths.multiply(mpi_85_mean,-1.0),'r--')
# # qplt.plot(mpi_45_mean,'g--')
# qplt.plot(iris.analysis.maths.multiply(mpi_26_mean,-1.0),'b--')
# plt.title('DMS flux')
# plt.savefig('/home/ph290/Documents/figures/dms_flux_ts.png')
# # plt.show()        


# plt.figure()
# qplt.plot(hadgem_85_mean_conc,'r')
# #qplt.plot(hadgem_45_mean,'g')
# qplt.plot(hadgem_26_mean_conc,'b')
# qplt.plot(mpi_85_mean_conc,'r--')
# # qplt.plot(mpi_45_mean,'g--')
# qplt.plot(mpi_26_mean_conc,'b--')
# plt.title('DMS conc.')
# plt.savefig('/home/ph290/Documents/figures/dms_conc_ts.png')
# # plt.show()                      


# '''
# MPI
# '''

# levels = np.linspace(0,5.0e-10,60)
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# levels2 = np.linspace(-1.0e-10,1.0e-10,60)
# norm2 = BoundaryNorm(levels2, ncolors=cmap.N, clip=True)


# cube = mpi_85.copy().collapsed('time',iris.analysis.MEAN)
# cube1 = mpi_85[0:19].copy().collapsed('time',iris.analysis.MEAN)
# cube_tmp1 = mpi_85[-19:-1].copy().collapsed('time',iris.analysis.MEAN)
# cube_tmp2 = mpi_85[0:19].copy().collapsed('time',iris.analysis.MEAN)
# cube2 = iris.analysis.maths.subtract(cube_tmp1,cube_tmp2)
# # 
# data1 = cube1.data
# data2 = cube2.data
# lats = cube1.coord('latitude').points
# lons = cube1.coord('longitude').points


# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# # ax = plt.subplot(211, projection=ccrs.PlateCarree())
# cax= plt.pcolormesh(lons, lats, data1*-1, cmap=cmap, norm=norm,
#              transform=ccrs.PlateCarree())
# ax.coastlines()
# ax.add_feature(cfeature.LAND)
# cbar = plt.colorbar(cax, orientation='horizontal')
# cbar.set_label('mol m$^{-2}$ s$^{-1}$')
# plt.title('DMS flux MPI-ESM-LR 2005-2025')
# plt.savefig('/home/ph290/Documents/figures/mpi_rcp85_dms_20yr_avg.png')
# # plt.show()

# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# # ax = plt.subplot(211, projection=ccrs.PlateCarree())
# cax = plt.pcolormesh(lons, lats, data2*-1, cmap=cmap2, norm=norm2,
#              transform=ccrs.PlateCarree())
# ax.coastlines()
# ax.add_feature(cfeature.LAND)
# cbar = plt.colorbar(cax, orientation='horizontal')
# cbar.set_label('mol m$^{-2}$ s$^{-1}$')
# plt.title('DMS flux MPI-ESM-LR 21stC change')
# plt.savefig('/home/ph290/Documents/figures/mpi_rcp85_dms_21stC_change.png')
# # plt.show()

# '''
# hadgem2-es
# '''

# cube = hadgem_85.copy().collapsed('time',iris.analysis.MEAN)
# cube1 = hadgem_85[0:19].copy().collapsed('time',iris.analysis.MEAN)
# cube_tmp1 = hadgem_85[-19:-1].copy().collapsed('time',iris.analysis.MEAN)
# cube_tmp2 = hadgem_85[0:19].copy().collapsed('time',iris.analysis.MEAN)
# cube2 = iris.analysis.maths.subtract(cube_tmp1,cube_tmp2)
# # 
# data1 = cube1.data
# data2 = cube2.data
# lats = cube1.coord('latitude').points
# lons = cube1.coord('longitude').points


# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# # ax = plt.subplot(211, projection=ccrs.PlateCarree())
# cax= plt.pcolormesh(lons, lats, data1, cmap=cmap, norm=norm,
#              transform=ccrs.PlateCarree())
# ax.coastlines()
# ax.add_feature(cfeature.LAND)
# cbar = plt.colorbar(cax, orientation='horizontal')
# cbar.set_label('mol m$^{-2}$ s$^{-1}$')
# plt.title('DMS flux HadGEM2-ES 2005-2025')
# plt.savefig('/home/ph290/Documents/figures/hadgem_rcp85_dms_20yr_avg.png')
# # plt.show()

# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# # ax = plt.subplot(211, projection=ccrs.PlateCarree())
# cax = plt.pcolormesh(lons, lats, data2, cmap=cmap2, norm=norm2,
#              transform=ccrs.PlateCarree())
# ax.coastlines()
# ax.add_feature(cfeature.LAND)
# cbar = plt.colorbar(cax, orientation='horizontal')
# cbar.set_label('mol m$^{-2}$ s$^{-1}$')
# plt.title('DMS flux HadGEM2-ES 21stC change')
# plt.savefig('/home/ph290/Documents/figures/hadgem_rcp85_dms_21stC_change.png')
# # plt.show()

# lana_data1 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_JAN.csv',delimiter=',')
# lana_data2 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_FEB.csv',delimiter=',')
# lana_data3 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_MAR.csv',delimiter=',')
# lana_data4 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_APR.csv',delimiter=',')
# lana_data5 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_MAY.csv',delimiter=',')
# lana_data6 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_JUN.csv',delimiter=',')
# lana_data7 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_JUL.csv',delimiter=',')
# lana_data8 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_AUG.csv',delimiter=',')
# lana_data9 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_SEP.csv',delimiter=',')
# lana_data10 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_OCT.csv',delimiter=',')
# lana_data11 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_NOV.csv',delimiter=',')
# lana_data12 = np.genfromtxt('/home/ph290/data0/observations/lana_dms/DMSclim_DEC.csv',delimiter=',')

# lana_data = np.mean([lana_data1,lana_data2,lana_data3,lana_data4,lana_data5,lana_data6,lana_data7,lana_data8,lana_data9,lana_data10,lana_data11,lana_data12],axis = 0)

# # plt.figure()
# # plt.contourf(lana_data,20)
# # plt.show()

# levels = np.linspace(0,12,60)

# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())

# cax = plt.contourf(np.linspace(-179.5,179.5,360), np.linspace(-89.5,89.5,180), np.flipud(lana_data),levels,
#              transform=ccrs.PlateCarree(),cmap=cmap)
# ax.coastlines()
# ax.add_feature(cfeature.LAND)
# cbar = plt.colorbar(cax, orientation='horizontal')
# cbar.set_label('nM')
# plt.title('DMS conc. climatology (Lana et al., 2011)')
# plt.savefig('/home/ph290/Documents/figures/lana_dms.png')
# # plt.show()

# '''
# model dms conc
# '''
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# data = hadgem_1985_2005_avg.data*1.0e6
# lats = hadgem_1985_2005_avg.coord('latitude').points
# lons = hadgem_1985_2005_avg.coord('longitude').points

# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cax = plt.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
#              transform=ccrs.PlateCarree())
# ax.coastlines()
# ax.add_feature(cfeature.LAND)
# cbar = plt.colorbar(cax, orientation='horizontal')
# cbar.set_label('nM')
# plt.title('DMS conc. HadGEM2-ES 1985-2005')
# plt.savefig('/home/ph290/Documents/figures/hadgem_1985_2005_conc.png')



# data = mpi_1985_2005_avg.data*1.0e6
# lats = mpi_1985_2005_avg.coord('latitude').points
# lons = mpi_1985_2005_avg.coord('longitude').points

# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cax = plt.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
#              transform=ccrs.PlateCarree())
# ax.coastlines()
# ax.add_feature(cfeature.LAND)
# cbar = plt.colorbar(cax, orientation='horizontal')
# cbar.set_label('nM')
# plt.title('DMS conc. MPI-ESM-LR 1985-2005')
# plt.savefig('/home/ph290/Documents/figures/mpi_1985_2005_conc.png')


'''
just downloaded historical and rcp85 concentrations to here for comparison with Lana, and for comparision with change (plotted above) to understand how much is just wind driven
/home/ph290/data0/cmip5_data/dms_conc/mpiesmlr/rcp85
'''

lnwdth = 2.5

plt.figure()
p1 = qplt.plot(hadgem_85_mean[1::],'r',linewidth = lnwdth)
# qplt.plot(hadgem_45_mean,'g')
p2 = qplt.plot(hadgem_26_mean[1::],'b',linewidth = lnwdth)
p3 = qplt.plot(iris.analysis.maths.multiply(mpi_85_mean,-1.0),'r--',linewidth = lnwdth)
# qplt.plot(mpi_45_mean,'g--')
p4 = qplt.plot(iris.analysis.maths.multiply(mpi_26_mean,-1.0),'b--',linewidth = lnwdth)
plt.title('DMS flux')
plt.ylabel('M m$^{-2}$ s$^{-1}$')
plt.xlabel('year')
# plt.legend([p1, p2, p3, p4], ["HadGEM RCP8.5", "HadGEM RCP2.6", "MPI RCP8.5","MPI RCP2.6"])
plt.savefig('/home/ph290/Documents/figures/dms_flux_ts.png')
# plt.show()        


plt.figure()
qplt.plot(hadgem_85_mean_conc[1::]*1.0e6,'r',linewidth = lnwdth)
#qplt.plot(hadgem_45_mean,'g')
qplt.plot(hadgem_26_mean_conc[1::]*1.0e6,'b',linewidth = lnwdth)
qplt.plot(mpi_85_mean_conc*1.0e6,'r--',linewidth = lnwdth)
# qplt.plot(mpi_45_mean,'g--')
qplt.plot(mpi_26_mean_conc*1.0e6,'b--',linewidth = lnwdth)
plt.title('DMS conc.')
plt.ylim(1.5,2.3)
plt.ylabel('nM')
plt.xlabel('year')
plt.savefig('/home/ph290/Documents/figures/dms_conc_ts.png')
# plt.show()  
