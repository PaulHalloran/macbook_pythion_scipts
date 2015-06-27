import numpy as np
import iris

def cube_extract_region(cube,min_lat,min_lon,max_lat,max_lon):
    n_cube_dims = cube.ndim
    cube_shape = cube.shape
    no_fields = 1
    if n_cube_dims == 2:
        mdi=cube.data.fill_value
    if n_cube_dims == 3:
        mdi=cube[0].data.fill_value
        no_fields = cube_shape[0]
    if n_cube_dims > 3:
        print 'greater than three dimensions - cant do this'
    if n_cube_dims < 2:
        print 'less than dimensions - cant do this'

    tmp=np.where((cube.coord('latitude').points <=  min_lat) | (cube.coord('longitude').points <= min_lon)  | (cube.coord('latitude').points >  max_lat) | (cube.coord('longitude').points > max_lon))
   
    new_cube=[]
    for i in range(no_fields):
        cube.data[i,tmp[0],tmp[1]]=mdi
	cube.data.mask[i,tmp[0],tmp[1]]=True

    return cube
