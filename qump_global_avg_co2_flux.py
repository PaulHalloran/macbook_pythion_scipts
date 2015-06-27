import iris
import numpy as np
import matplotlib.pyplot as plt
import glob
import iris.analysis.cartography
import iris.coord_categorisation
import iris.analysis
import time
import matplotlib as mpl

#time.sleep(60.0*60.0*12*2)

def my_callback(cube, field, filename):
        cube.remove_coord('forecast_reference_time')
        cube.remove_coord('forecast_period')
        #the cubes were not merging properly before, because the time coordinate appeard to have teo different names... I think this may work

directory = '/data/data1/ph290/qump_co2/stash_split/qump_n_atl_mor_var_monthly_ss/'
output_directory = ('/home/ph290/data1/qump_co2/global_avg/')

runs = glob.glob(directory+'/?????')

run_names = []
run_global_means = []
run_date = []

for i,run in enumerate(runs):
    if i >= 114:
        print i
        run_name = run.split('/')[7]
        run_names.append(run_name)
        cube = iris.load_cube(run+'/*02.30.249*.pp',iris.AttributeConstraint(STASH='m02s30i249'),callback=my_callback)
        iris.coord_categorisation.add_year(cube, 'time', name='year2')
        cube2 = cube.aggregated_by('year2', iris.analysis.MEAN)
        cube2.coord('longitude').guess_bounds()
        cube2.coord('latitude').guess_bounds()
        grid_areas = iris.analysis.cartography.area_weights(cube2)
        time_mean = cube2.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
        run_global_means.append(time_mean.data)
        coord = cube2.coord('time')
        year = np.array([coord.units.num2date(value).year for value in coord.points])
        run_date.append(year)
        np.savetxt(output_directory + run_name + '.txt',np.vstack((year,time_mean.data)).T,delimiter=',')

fig = plt.figure()
for i,data in enumerate(run_global_means):
    if np.size(np.where(data.data <= -0.5)) == 0:
	if np.size(np.where(data[1::]-data[0:-1] > 0.5)) == 0:
            plt.plot(run_date[i],data,'k')
            plt.xlabel('year')
            plt.ylabel('air-sea CO$_2$ flux anomaly (mol-C m$^{-2}$ yr$^{-1}$)')

mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8
plt.show()



