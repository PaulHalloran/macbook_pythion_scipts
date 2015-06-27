import time
import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess
import os
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy.ma as ma
import running_mean
from scipy import signal
import scipy
import scipy.stats
import numpy as np
import statsmodels.api as sm
import running_mean_post
from scipy.interpolate import interp1d


#this is a simple function that we call later to look at the file names and extarct from them a unique list of models to process
#note that the model name is in the filename when downlaode ddirectly from the CMIP5 archive
def model_names(directory,variable):
	files = glob.glob(directory+'/*'+variable+'*.nc')
	models_tmp = []
	for file in files:
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[0])
			models = np.unique(models_tmp)
	return models


first process: /data/temp/ph290/last_1000

input_directory = '/media/usb_external1/cmip5/reynolds_data/'

variables = np.array(['rsds'])
#specify the temperal averaging period of the data in your files e.g for an ocean file 'Omon' (Ocean monthly) or 'Oyr' (ocean yearly). Atmosphere would be comething like 'Amon'. Note this just prevents probvlems if you accidently did not specify the time frequency when doenloading the data, so avoids trying to puut (e.g.) daily data and monthly data in the same file.

'''
#Main bit of code follows...
'''



models = model_names(input_directory,variables[0])

models = list(models)
#models.remove('BNU-ESM')
#models.remove('CanESM2')
#models.remove('MIROC5')
#models.remove('MRI-CGCM3')
#models.remove('CSIRO-Mk3-6-0')
import running_mean

cube1 = iris.load_cube(input_directory+models[0]+'*'+variables[0]+'*.nc')[0]

models2 = []
cubes = []

for model in models:
	print 'processing: '+model
	try:
		cube = iris.load_cube(input_directory+model+'*'+variables[0]+'*.nc')
	except:
		cube = iris.load(input_directory+model+'*'+variables[0]+'*.nc')
		cube = cube[0]
	cubes.append(cube)


coord = cubes[0].coord('time')
dt = coord.units.num2date(coord.points)
model_years = np.array([coord.units.num2date(value).year for value in coord.points])


'''
Volcanic forcing
'''

file1 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090N_AOD_c.txt'
file2 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030N_AOD_c.txt'
file3 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090S_AOD_c.txt'
file4 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030S_AOD_c.txt'

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)
data3 = np.genfromtxt(file2)
data4 = np.genfromtxt(file3)
data_tmp = np.zeros([data1.shape[0],2])
data_tmp[:,0] = data1[:,1]
data_tmp[:,1] = data2[:,2]
data = np.mean(data_tmp,axis = 1)
data_final = data1.copy()
data_final[:,1] = data

volcanic_smoothing = 65 #yrs

volc_years = data1[:,0]

loc = np.where((volc_years >= np.min(model_years)) & (volc_years <= np.min(model_years)))

volc = running_mean_post.running_mean_post(data_final_t[loc],12.0*volcanic_smoothing)
volc_mean = np.mean(volc)
high_volc = np.where(volc > volc_mean)
low_volc = np.where(volc < volc_mean)

high_volc_data = np.empty([np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1],np.size(cubes)])
low_volc_data = high_volc_data.copy()

for i,cube in enumerate(cubes):
	tmp = cube[high_volc].collapsed('time',iris.analysis.MEAN)
	high_volc_data[i,:,:] = tmp.data
	tmp = cube[low_volc].collapsed('time',iris.analysis.MEAN)
	low_volc_data[i,:,:] = tmp.data

high_volc_data_mean = np.mean(high_volc_data,axis = 0)
low_volc_data_mean = np.mean(low_volc_data,axis = 0)

high_volc_data_mean_cube = cubes[0][0].copy()
high_volc_data_mean_cube.data = high_volc_data_mean
low = cubes[0][0].copy()
low_volc_data_mean_cube.data = low_volc_data_mean

plt.figure()
qplt.contourf(high_volc_data_mean_cube,30)
plt.show()


plt.figure()
qplt.contourf(low_volc_data_mean_cube,30)
plt.show()








