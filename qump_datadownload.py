import glob
import iris
import subprocess
import numpy as np

f = open('my_query_file','w')
f.write('begin\n')
f.write('stash=(00103,30249)\n')
f.write('end\n')
f.close()

run_bases  = ['anun','anuq','anur','anuv']
last_letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']

subprocess.call(['mkdir qump_emulation_data'], shell=True)

for run_base in run_bases:
	for last_letter in last_letters:
		subprocess.call(['moo select -C ./my_query_file moose:crum/'+run_base+last_letter+'/opy.pp ./qump_emulation_data'+run_base+last_letter+'.pp'], shell=True)

run_names = glob.glob('./qump_emulation_data/*.pp')


for run_name in run_names:
	air_sea_flux_stash = iris.AttributeConstraint(STASH='m02s30i249'))
	#Read in air-sea flux
	cube = iris.load_cube('./qump_emulation_data'+run_name,air_sea_flux_stash)
	#extract time data
	coord = cube.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])
	#area average air-sea flux
	cube.coord('latitude').guess_bounds()
	cube.coord('longitude').guess_bounds()
	grid_areas = iris.analysis.cartography.area_weights(cube)
	area_avged_cube = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
	f = open('area_avged_air_sea_flux.txt','w')
	f.write(run_name+'\n')	
	f.write(np.str(year)+'\n')
	f.write(np.str(area_avged_cube.data)+'\n')
	f.close()
	
for run_name in run_names:
	tco2_stash = iris.AttributeConstraint(STASH='m02s00i103'))
	#Read in air-sea flux
	cube = iris.load_cube('./qump_emulation_data'+run_name,tco2_stash)
	#extract time data
	coord = cube.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])
	#area average air-sea flux
	cube.coord('latitude').guess_bounds()
	cube.coord('longitude').guess_bounds()
	cube.coord('depth').guess_bounds()
	grid_areas = iris.analysis.cartography.area_weights(cube)
	vol_avged_cube = cube.collapsed(['longitude', 'latitude','depth'], iris.analysis.MEAN, weights=grid_areas)
	f = open('vol_avged_tco2.txt','w')
	f.write(run_name+'\n')	
	f.write(np.str(year)+'\n')
	f.write(np.str(vol_avged_cube.data)+'\n')
	f.close()
	
	