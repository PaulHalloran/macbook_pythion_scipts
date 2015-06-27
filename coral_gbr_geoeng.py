import iris
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import itertools
import numpy
import iris.plot as iplt

def my_callback(cube, field, filename):
    if field.lbuser[3] != 101:
        raise iris.exceptions.IgnoreCubeException()

filename='/data/local/hador/mass_retrievals/coral_geo/amzzb/*.pp'
temp_cubes = iris.load(filename, callback=my_callback)

mean_data = temp_cubes[0].collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
ticks = (mean_data.coord("time").points/24.0/360.0)+1970.0


GBR = iris.Constraint(
                                    longitude=lambda v: 143 <= v <= 147,
                                    latitude=lambda v: -25 <= v <= -10,
                                    name='sea_water_potential_temperature'
                                    )
Globe= iris.Constraint(
                                    longitude=lambda v: 0 <= v <= 360,
                                    latitude=lambda v: -90 <= v <= 90,
                                    name='sea_water_potential_temperature'
                                    )

GBR_data=temp_cubes[0].extract(GBR)
GBR_mean=GBR_data.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)

Global_data=temp_cubes[0].extract(Globe)
Global_mean=Global_data.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)

contour_levels = numpy.linspace(10, 30, 30)

for i,temp_slice in enumerate(GBR_data.slices(['longitude', 'latitude'])):
#loop over years to do multiple plots
    plt.subplot(4, 5, i)
    cf = iplt.contourf(temp_slice, contour_levels)
    plt.gca().coastlines()
    plt.suptitle(ticks[i])


plt.show()


#results from amzzb:[27.0210629872 26.9715627489 27.1222252619 27.2693071365 25.3531584058
# 23.2523442677 22.0357715062 21.4674273445 21.1954474676 21.0140721457
# 20.7787440618 20.4177302406 20.2809268861 19.8982352756 20.2288796107
# 20.2069899241 19.9980516888 19.463846025 19.3741356986 19.1533955165]


#results from amzzc:[25.4595063073 25.2209613437 24.9404356366 24.7648945763 22.6167292822
# 20.7128760928 19.3539028622 19.5456107003 19.756918453 18.6241522744
# 18.3678275517 18.0327098483 18.665843782 18.6083708718 18.7510402316
# 18.2124869937 18.3569473539 18.017865953 17.887540454 18.1281205132]

#But note that at the same time, we cool the world globally by A LOT!
#[16.4109311566 16.2077393311 16.0334892339 15.773384555 14.3665036121
# 12.9771760885 12.292647734 11.9147016889 11.4396537529 11.1160967637
# 10.9424512178 10.8467402354 10.7849326522 10.6666130211 10.5287188784
# 10.3811292321 10.343834523 10.2279784897 10.0264254828 9.88865462475]

#it looks to me like the co2 is back bown at preindustrial or somethiong - this is clearly not a normal climate run, and it is not the so2 causing the prioblem - you get he same whatever the emission

