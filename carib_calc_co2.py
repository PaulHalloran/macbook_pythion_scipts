import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np
import iris.analysis
import statsmodels.api as sm

qlodap_dir = '/home/ph290/data1/observations/glodap/'
woa_dir = '/home/ph290/Documents/teaching/'

tco2_in = iris.load_cube(qlodap_dir+'TCO2.nc','Total_CO2')

alk_in = iris.load_cube(qlodap_dir+'PALK.nc','Potential_Alkalinity')

sst_in = iris.load_cube(woa_dir+'temperature_annual_1deg.nc','sea_water_temperature')

sss_in = iris.load_cube(woa_dir+'salinity_annual_1deg.nc','sea_water_salinity')

depth_levs = tco2_in.shape[0]

mlr_alk = sst_in[0].copy()
mlr_alk.data[:] = np.nan
mlr_tco2 = sst_in[0].copy()
mlr_tco2.data[:] = np.nan

pco2 = sst_in[0].copy()
pco2_delta_calc = sst_in[0].copy()

basin_mask = iris.load_cube(woa_dir+'basin.nc')
basin_mask = basin_mask[0][0]
basin_mask.data[np.where(np.logical_not(basin_mask.data == 1))] = np.nan
basin_mask_flipped = iris.analysis.maths.np.flipud(basin_mask.data)

mlr_tco2_tmp = mlr_tco2.data
mlr_alk_tmp = mlr_alk.data


for i in np.arange(depth_levs):

    alk = alk_in[i]
    tco2 = tco2_in[i]
    sst =sst_in[0][i]
    sss = sss_in[0][i]

    basin_mask = iris.load_cube(woa_dir+'basin.nc')
    basin_mask = basin_mask[0][0]
    basin_mask.data[np.where(np.logical_not(basin_mask.data == 1))] = np.nan
    basin_mask_flipped = iris.analysis.maths.np.flipud(basin_mask.data)

    sstb = iris.analysis.maths.multiply(sst,basin_mask_flipped)
    sssb = iris.analysis.maths.multiply(sss,basin_mask_flipped)
    tco2b = iris.analysis.maths.multiply(tco2,np.roll(np.flipud(np.rot90(basin_mask_flipped)),180,0))
    alkb = iris.analysis.maths.multiply(alk,np.roll(np.flipud(np.rot90(basin_mask_flipped)),180,0))

    tco2b.transpose()
    alkb.transpose()

    glodap_mask = tco2b.copy()
    glodap_mask.data[np.where(tco2b.data > 1)] = 1

    sstc = iris.analysis.maths.multiply(sstb,np.roll(glodap_mask.data,180,1))
    sssc = iris.analysis.maths.multiply(sssb,np.roll(glodap_mask.data,180,1))

    woa_mask = sstc.copy()
    woa_mask.data[np.where(sstc.data > 1)] = 1
    alkc = iris.analysis.maths.multiply(alkb,np.roll(woa_mask.data,180,1))

    '''
    now try and calculate what alkalnity may be like in the caribbean...
    '''

    '''
    initially doing this by working out the atlantic multiple linear regression between salinity, temperature and alkalinity
    '''

    alkc.data[alkc.data.mask] = np.nan
    sstc.data[sstc.data.mask] = np.nan
    sssc.data[sssc.data.mask] = np.nan

    x1 = np.reshape(sstc.data,180*360)
    x2 = np.reshape(sssc.data,180*360)
    y = np.reshape(np.roll(alkc.data,180,1),180*360)

    x1b = x1[(np.logical_not(np.isnan(x1)))]
    x2b = x2[(np.logical_not(np.isnan(x2)))]
    yb = y[(np.logical_not(np.isnan(x2)))]

    x1c = x1b[(np.logical_not(np.isnan(yb)))]
    x2c = x2b[(np.logical_not(np.isnan(yb)))]
    yc = yb[(np.logical_not(np.isnan(yb)))]

    x = np.column_stack((x1c,x2c))
    x = sm.add_constant(x)
    model = sm.OLS(yc,x)
    results = model.fit()
    print results.summary()

    #plt.plot(yc)
    #plt.plot(results.params[2]*x2c+results.params[1]*x1c+results.params[0])
    #plt.show()
    #R-squared:                       0.733

    #note, if we remove sst from the MLR, we significantly reduce the R2
    #x = sm.add_constant(x2c)
    #model = sm.OLS(yc,x)
    #results = model.fit()
    #R-squared:                       0.571


    '''
    now use this relationship to try and come up with an alkalinity map infilled for the Caribbean
    '''

    mlr_alk_tmp[i,:,:] = (sssb.data * results.params[2]) + (sstb.data * results.params[1]) + results.params[0]
    #plt.contourf((sssb.data * results.params[2]) + (sstb.data * results.params[1]) + results.params[0],np.linspace(2000,2500,30))
    #plt.show()


    '''
    and the same for TCO2 - althoug hit woul dbe better to use takahashi 2009 CO2 values, and work back, rather than using DIC...
    '''

    tco2c = iris.analysis.maths.multiply(tco2b,np.roll(woa_mask.data,180,1))

    x1 = np.reshape(sstc.data,180*360)
    x2 = np.reshape(sssc.data,180*360)
    y = np.reshape(np.roll(tco2c.data,180,1),180*360)

    x1b = x1[(np.logical_not(np.isnan(x1)))]
    x2b = x2[(np.logical_not(np.isnan(x2)))]
    yb = y[(np.logical_not(np.isnan(x2)))]

    x1c = x1b[(np.logical_not(np.isnan(yb)))]
    x2c = x2b[(np.logical_not(np.isnan(yb)))]
    yc = yb[(np.logical_not(np.isnan(yb)))]


    x = np.column_stack((x1c,x2c))
    x = sm.add_constant(x)
    model2 = sm.OLS(yc,x)
    results2 = model2.fit()

    '''
    now use this relationship to try and come up with an tco2 map infilled for the Caribbean
    '''

    mlr_tco2_tmp[i,:,:] = (sssb.data * results2.params[2]) + (sstb.data * results2.params[1]) + results2.params[0]

    # qplt.contourf(mlr_tco2[i],np.linspace(1926,2100,30))
    # plt.show()

    # qplt.contourf(tco2b,np.linspace(1926,2100,30))
    # plt.show()

mlr_tco2.data = mlr_tco2_tmp
mlr_alk.data = mlr_alk_tmp

import carbchem


pco2_tmp = pco2.data
pH_tmp = pco2.data
for i in np.arange(pco2_tmp.shape[0]):
    tco2_in2 = tco2_in[i]
    tco2_in2.transpose()
    alk_in2 = alk_in[i]
    alk_in2.transpose()
    pco2_tmp[i,:,:] = carbchem.carbchem(1,0.0,sst_in[0][i].data,sss_in[0][i].data,mlr_tco2_tmp[i,:,:]/(1026.*1000.),mlr_alk_tmp[i,:,:]/(1026.*1000.))
    pH_tmp[i,:,:] = carbchem.carbchem(2,0.0,sst_in[0][i].data,sss_in[0][i].data,mlr_tco2_tmp[i,:,:]/(1026.*1000.),mlr_alk_tmp[i,:,:]/(1026.*1000.))

    #CARBCHEM.carbchem(1,0.0,sst_in[0][i].data,sss_in[0][i].data,tco2_in2.data/(1000.*1000.),alk_in2.data/(1000.*1000.))

#note that his is only applicable near the surface because I've not introducsed the pressure term...

#note that the pCO2 we calculate using glodap and woa is high when before I start MLRing. Why is this? Glodap is paracticle alkalinity, as the carbem script uses.

pco2.data = pco2_tmp
pH = pco2
pH.data = pH_tmp

qplt.contourf(pco2[0])
plt.show()

'''
Now calculate what diffreence emma's extra total alkalinity makes to pCO2
'''
#CaCO3 has a molecular weight of 100 g/mol 
#The CO3 anion has a molecular weight of 60 g/mol 
#1 mol CaCO3 reduces DIC = CO2 + HCO + CO2 by 1 mol
#1 mol CaCO3 reduces alkalinity = HCO + 2 CO2 + . . . by 2 mol
#area of caribbean = 2754000*1000000 m2

#0.19 Gt CaCO3 was being produced in 1970s not being produced now

g_caco3 = 0.19*1.0e9*1.0e6 #g caco3 per year
over_depth_of = 20
g_per_m2 = (g_caco3/(2754000*1000000))/over_depth_of
#and diluted over 20m depth
mol_per_kg = g_per_m2/(100.0*1026.0)



pco2_tmp2 = pco2.data*0.0
pH_tmp2 = pH_tmp.data*0.0
for i in np.arange(pco2_tmp.shape[0]):
    tco2_in2 = tco2_in[i]
    tco2_in2.transpose()
    alk_in2 = alk_in[i]
    alk_in2.transpose()
    mlr_tco2_tmp2 = mlr_tco2_tmp[i,:,:]/(1026.*1000.)
    mlr_tco2_tmp3 = mlr_tco2_tmp2+mol_per_kg
    mlr_alk_tmp2 = mlr_alk_tmp[i,:,:]/(1026.*1000.)
    mlr_alk_tmp3 = mlr_alk_tmp2 + (mol_per_kg*2.0)
    pco2_tmp2[i,:,:] = carbchem.carbchem(1,0.0,sst_in[0][i].data,sss_in[0][i].data,mlr_tco2_tmp3,mlr_alk_tmp3)
    pH_tmp2[i,:,:] = carbchem.carbchem(2,0.0,sst_in[0][i].data,sss_in[0][i].data,mlr_tco2_tmp3,mlr_alk_tmp3)

pco2_delta_calc.data = pco2_tmp2

# pco2_tmp2 = pco2.data*0.0
# pco2_tmp2[:,:] = carbchem.carbchem(1,0.0,sst_in[0][0].data,sss_in[0][0].data,mlr_tco2_tmp3,mlr_alk_tmp3)

diff = pco2.copy()
diff.data = diff.data*np.nan
diff.data = pco2_tmp2-pco2_tmp[0,:,:]
diff.standard_name = 'surface_partial_pressure_of_carbon_dioxide_in_sea_water'
diff.units = 'uatm'

qplt.contourf(diff[0],50)
plt.gca().coastlines()
plt.show()

pH_diff = pco2.copy()
pH_diff.data = pH_diff.data*np.nan
pH_diff.data = pH_tmp2-pH_tmp[0,:,:]
pH_diff.standard_name = 'sea_water_ph_reported_on_total_scale'
pH_diff.units = '1'

qplt.contourf(pH_diff[0],50)
plt.gca().coastlines()
plt.show()


'''
depth profiles (for interest)
'''


# alk_in = iris.load_cube(qlodap_dir+'PALK.nc','Potential_Alkalinity')
# sss_in = iris.load_cube(woa_dir+'salinity_annual_1deg.nc','sea_water_salinity')
# sss_in = sss_in[0]



# alk_in.coord('latitude').guess_bounds()
# alk_in.coord('longitude').guess_bounds()
# grid_areas = iris.analysis.cartography.area_weights(alk_in)
# alk_in_profile = alk_in.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights=grid_areas)

# grid_areas = iris.analysis.cartography.area_weights(sss_in)
# sss_in_profile = sss_in.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights=grid_areas)

# qplt.plot(iris.analysis.maths.divide(alk_in_profile,sss_in_profile))
# plt.show()


# extract_area = iris.Constraint(longitude=lambda v: -60 <= v <= 10,latitude=lambda v: 0 <= v <= 60)
# alk_in_extracted = alk_in.extract(extract_area)
# sss_in_extracted = sss_in.extract(extract_area)

# grid_areas = iris.analysis.cartography.area_weights(alk_in_extracted)
# alk_in_profile = alk_in_extracted.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights=grid_areas)

# grid_areas = iris.analysis.cartography.area_weights(sss_in_extracted)
# sss_in_profile = sss_in_extracted.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights=grid_areas)

# qplt.plot(iris.analysis.maths.divide(alk_in_profile,sss_in_profile))
# plt.show()
