import numpy as np
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt

'''
Read in HadSST3 data, whcih does not have any satilite information (only in situe obs)
'''

#file = '/home/ph290/data1/HadSST.3.1.0.0.median.nc'
file = '/home/ph290/data1/HadSST2_1850on.nc'

#cube=iris.load(file,'SST anomaly')[0]

cube=iris.load(file,'Monthly 5degree resolution SST anomalies wrt 1961-90 climatology')[0]

coord = cube.coord('time')
dt = coord.units.num2date(coord.points)

yr=[]
mn=[]
dy=[]
for i in range(dt.size):
    yr.append(dt[i].year)
    mn.append(dt[i].month)
    dy.append(dt[i].day)

cube.data[np.where(cube.data.mask == True)] = np.nan

'''
Read in HadISST data for comparision
'''

# file2 = '/home/ph290/data1/HadISST_sst.nc'
# cube2=iris.load(file2,'sea_surface_temperature')
# cube2=cube2[0]

# coord2 = cube2.coord('time')
# dt2 = coord2.units.num2date(coord2.points)

# yr2=[]
# mn2=[]
# dy2=[]
# for i in range(dt2.size):
#     yr2.append(dt[i].year)
#     mn2.append(dt[i].month)
#     dy2.append(dt[i].day)

# cube2.data[np.where(cube2.data.mask == True)] = np.nan
# hadisst_on_hadsst2_grid = iris.analysis.interpolate.regrid(cube2, cube[0], mode='bilinear')
# tmp_shape=hadisst_on_hadsst2_grid.data.shape
# cube_b = iris.analysis.interpolate.regrid(cube, cube[0], mode='bilinear')
# cube_c=cube_b[0:tmp_shape[0]]


'''
ensure that hadsst3 has hadisst mask
'''

# cube_c.data[np.where(np.isnan(hadisst_on_hadsst2_grid.data))] = np.nan

# coord = cube_c.coord('time')
# dt = coord.units.num2date(coord.points)

# yr=[]
# mn=[]
# dy=[]
# for i in range(dt.size):
#     yr.append(dt[i].year)
#     mn.append(dt[i].month)
#     dy.append(dt[i].day)
    

'''
Count number of empty grid cells in each setup for each month (the difference will be the number of misising 5x5 cells with obs - because they will both have the same misisng data points for land)
'''

regions=['Greater Caribbean Region','South East Asia','Central Pacific','West Indian Ocean','Great Barrier Reef and Polynesia']
lon_west=[-84,96,-180,29,142]
lon_east=[-53,127,-149,60,173]
lat_south=[9,-11,-10,-31,-25]
lat_north=[40,20,21,0,6]

for j in range(np.size(regions)):

    region = iris.Constraint(longitude=lambda v: lon_west[j] <= v <= lon_east[j],latitude=lambda v: lat_south[j] <= v <= lat_north[j])
    cube_region = cube.extract(region)
    #if you want to compare with hadisst use cube_region = cube_c.extract(region)
    #cube_region2 = hadisst_on_hadsst2_grid.extract(region)

    mdi_squares=[]
    mdi_squares2=[]
    for i in range(dt.size):
            try:
                count=np.size(np.where(np.isnan(cube_region[i].data))[0])
                # print count
                mdi_squares.append(count)
            except:
                mdi_squares.append(0)

    # for i in range(dt2.size):
    #         try:
    #             count=np.size(np.where(np.isnan(cube_region2[i].data))[0])
    #             mdi_squares2.append(count)
    #         except:
    #             mdi_squares2.append(0)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(dt,mdi_squares)
    # ax.plot(dt2,mdi_squares2,'b')
    ax.xaxis.set_ticks([dt[0],dt[12*50],dt[12*100],dt[12*150]])
    plt.xlabel('date')
    plt.ylabel('empty grid cells (including land cells)')
    plt.title(regions[j])
    plt.savefig('/home/ph290/Documents/figures/'+regions[j].replace(' ','')+'_empty_cells.ps')


# x=np.where(np.array(yr) == 1975)
# qplt.contourf(cube_region.extract(iris.Constraint(time = coord.points[x[0][0]])))
# plt.show()

# plt.figure()
# plt.subplot(2,1,1)
# qplt.contourf(cube_c[1500])
# plt.subplot(2,1,2)
# qplt.contourf(hadisst_on_hadsst2_grid[1500])
# plt.show()
