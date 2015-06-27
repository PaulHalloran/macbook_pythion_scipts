'''
USE Sally E. Close1 and Hugues Goosse1 JOURNAL OF GEOPHYSICAL RESEARCH: OCEANS, VOL 118 2811-2827 2013
'''

from iris.coords import DimCoord
import iris.plot as iplt
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
import running_mean as rm
import running_mean_post as rmp
from scipy import signal
import scipy
import scipy.stats
import numpy as np
import statsmodels.api as sm
import running_mean_post
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import iris.analysis.cartography
import numpy.ma as ma
import scipy.interpolate
import gc
import pickle
import biggus
import seawater
import cartopy.feature as cfeature
import statsmodels.api as sm
from eofs.iris import Eof
import cartopy.feature as cfeature
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
from scipy.stats import gaussian_kde
from statsmodels.stats.outliers_influence import summary_table
import scipy.ndimage
import scipy.ndimage.filters
import gsw



def mld(S,thetao,depth_cube,latitude_deg):
	"""Compute the mixed layer depth.
	Parameters
	----------
	SA : array_like
		 Absolute Salinity  [g/kg]
	CT : array_like
		 Conservative Temperature [:math:`^\circ` C (ITS-90)]
	p : array_like
		sea pressure [dbar]
	criterion : str, optional
			   MLD Criteria
	Mixed layer depth criteria are:
	'temperature' : Computed based on constant temperature difference
	criterion, CT(0) - T[mld] = 0.5 degree C.
	'density' : computed based on the constant potential density difference
	criterion, pd[0] - pd[mld] = 0.125 in sigma units.
	`pdvar` : computed based on variable potential density criterion
	pd[0] - pd[mld] = var(T[0], S[0]), where var is a variable potential
	density difference which corresponds to constant temperature difference of
	0.5 degree C.
	Returns
	-------
	MLD : array_like
		  Mixed layer depth
	idx_mld : bool array
			  Boolean array in the shape of p with MLD index.
	Examples
	--------
	>>> import os
	>>> import gsw
	>>> import matplotlib.pyplot as plt
	>>> from oceans import mld
	>>> from gsw.utilities import Bunch
	>>> # Read data file with check value profiles
	>>> datadir = os.path.join(os.path.dirname(gsw.utilities.__file__), 'data')
	>>> cv = Bunch(np.load(os.path.join(datadir, 'gsw_cv_v3_0.npz')))
	>>> SA, CT, p = (cv.SA_chck_cast[:, 0], cv.CT_chck_cast[:, 0],
	...              cv.p_chck_cast[:, 0])
	>>> fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharey=True)
	>>> l0 = ax0.plot(CT, -p, 'b.-')
	>>> MDL, idx = mld(SA, CT, p, criterion='temperature')
	>>> l1 = ax0.plot(CT[idx], -p[idx], 'ro')
	>>> l2 = ax1.plot(CT, -p, 'b.-')
	>>> MDL, idx = mld(SA, CT, p, criterion='density')
	>>> l3 = ax1.plot(CT[idx], -p[idx], 'ro')
	>>> l4 = ax2.plot(CT, -p, 'b.-')
	>>> MDL, idx = mld(SA, CT, p, criterion='pdvar')
	>>> l5 = ax2.plot(CT[idx], -p[idx], 'ro')
	>>> _ = ax2.set_ylim(-500, 0)
	References
	----------
	.. [1] Monterey, G., and S. Levitus, 1997: Seasonal variability of mixed
	layer depth for the World Ocean. NOAA Atlas, NESDIS 14, 100 pp.
	Washington, D.C.
	""" 
	#depth_cube.data = np.ma.masked_array(np.swapaxes(np.tile(depths,[360,180,1]),0,2))
	try:
		S.coord('depth')
		MLD_out = S.extract(iris.Constraint(depth = np.min(depth_cube.data)))
	except:
		MLD_out = S[:,0,:,:]
	MLD_out_data = MLD_out.data
	for i in range(np.shape(MLD_out)[0]):
		print'calculating mixed layer for year: ',i
		thetao_tmp = thetao[i]
		S_tmp = S[i]
		depth_cube.data = np.abs(depth_cube.data)
		depth_cube = depth_cube * (-1.0)
		p = gsw.p_from_z(depth_cube.data,latitude_deg.data) # dbar
		SA = S_tmp.data*1.004715
		CT = gsw.CT_from_pt(SA,thetao_tmp.data - 273.15)
		SA, CT, p = map(np.asanyarray, (SA, CT, p))
		SA, CT, p = np.broadcast_arrays(SA, CT, p)
		SA, CT, p = map(ma.masked_invalid, (SA, CT, p))
		p_min, idx = p.min(axis = 0), p.argmin(axis = 0)
		sigma = SA.copy()
		to_mask = np.where(sigma == S.data.fill_value)
		sigma = gsw.rho(SA, CT, p_min) - 1000.
		sigma[to_mask] = np.NAN
		sig_diff = sigma[0,:,:].copy()
		sig_diff += 0.125 # Levitus (1982) density criteria
		sig_diff = np.tile(sig_diff,[np.shape(sigma)[0],1,1])
		idx_mld = sigma <= sig_diff
		#NEED TO SORT THS PIT - COMPARE WWITH OTHER AND FIX!!!!!!!!!!
		MLD = ma.masked_all_like(S_tmp.data)
		MLD[idx_mld] = depth_cube.data[idx_mld] * -1
		MLD_out_data[i,:,:] = np.ma.max(MLD,axis=0) 
	return MLD_out_data



models = ['MRI-CGCM3','MPI-ESM-P', 'GISS-E2-R','CSIRO-Mk3L-1-2', 'HadCM3', 'CCSM4']
#,,'MIROC-ESM']
#NOTE MIROC currently crashes things (looks like a memory issue - segmentation sault) - find out why
#models = ['MPI-ESM-P', 'GISS-E2-R','CSIRO-Mk3L-1-2', 'HadCM3']
models = ['MPI-ESM-P']

directory = '/data/NAS-ph290/ph290/cmip5/last1000/'


######################################
#   Calculating Mixed layer depths   #
######################################

ensembles = ['r1i1p1','r1i1p121']

print 'Calculating mixed layer depths'


for ensemble in ensembles:
	for model in models:
		print model
		try:
			test = glob.glob(directory+model+'_my_mld_'+ensemble+'.nc')
			if np.size(test) == 0:
				print 'calculating MLD for '+model
				S = iris.load_cube(directory+model+'*_so_past1000_'+ensemble+'_*Omon*.nc') 
				thetao = iris.load_cube(directory+model+'*_thetao_past1000_'+ensemble+'_*Omon*.nc') 
				#S = S[0:2]
				#thetao = thetao[0:2]
				depth_cube = S[0].copy()
				try:
					S.coord('depth')
					depths = depth_cube.coord('depth').points
				except:
					depths = depth_cube.coord('ocean sigma over z coordinate').points
				 #for memorys sake, do one year at a time..
				depth_cube.data = np.ma.masked_array(np.swapaxes(np.tile(depths,[360,180,1]),0,2))
				try:
					S.coord('depth')
					hm = S.extract(iris.Constraint(depth = np.min(depth_cube.data)))
				except:
					hm = S[:,0,:,:]
				latitude_deg = depth_cube.copy()
				latitude_deg_data = hm.coord('latitude').points
				latitude_deg.data = np.swapaxes(np.tile(latitude_deg_data,[np.shape(S)[1],360,1]),1,2)
				hm.data = mld(S,thetao,depth_cube,latitude_deg)
				iris.fileformats.netcdf.save(hm,directory+model+'_my_mld_'+ensemble+'.nc')
			else:
				print 'MLD for '+model+' already exists'
		except:
			print ensemble+' '+model+' failed'
		
		
model_data = {}
ensemble = 'r1i1p1'

##############################
#     constants              #
##############################

#number of seconds in a year
yearsec = 60.0 * 60.0 * 24.0 * 360.0
# eddy diffusivity - held constant, following Dong et al. [2009],
k = 500 #m2/s
#  characteristic density of the mixed layer (taken here to be 1027kg m3)
po = 1027.0
# density of sea ice, estimated to be 930 kg m-3
pi = 930.0
# salinity of sea ice, estimated here as 5
Si = 5
#note that upside delta (nabia) is the gradient and can be calculated with np.gradient() but this is calculates in all three dimentsin (time, x and y), so to get what you waht use:
#tmp = np.gradient(variable_of_interest.data)
#gradient_of_variable.data = tmp[1]+tmp[2]
# nabia squared is called the laplacian and can be calculated with scipy.ndimage.filters.laplace() I think

##############################
#     salinity budget        #
##############################

print 'Calculating salinity budget'


for model in models:
	evap_precip_flag = False
	print model
	model_data[model] = {}
	######################################
	# Sea ice concentration              #
	######################################
	sic = iris.load_cube(directory+model+'*_sic_past1000_r1i1p1_*.nc')
	######################################
	# P minus E                          #
	######################################
	try:
		# mass of water vapor evaporating from the ice-free portion of the ocean
		E_no_ice = iris.load_cube(directory+model+'*_evs_past1000_r1i1p1_*Omon*.nc')
		# mass of liquid water falling as liquid rain  into the ice-free portion of the ocean
		P_no_ice_rain = iris.load_cube(directory+model+'*_pr_past1000_r1i1p1_*Omon*.nc')
		# mass of solid water falling as liquid rain  into the ice-free portion of the ocean
		P_no_ice_snow = iris.load_cube(directory+model+'*_prsn_past1000_r1i1p1_*Omon*.nc')
		P_no_ice = P_no_ice_rain + P_no_ice_snow
		P_minus_E_no_ice = P_no_ice - E_no_ice
		P_minus_E_no_ice *= yearsec
		#convert into a flux per year
	except:
		# mass of water vapor evaporating from all portion of the ocean
		E = iris.load_cube(directory+model+'*_evspsbl_past1000_r1i1p1_*.nc')
		# mass of solid+liquid water falling into the whole of the ocean
		P = iris.load_cube(directory+model+'*_pr_past1000_r1i1p1_*Amon*.nc')
		P_minus_E_no_ice = (P - E) * (((sic*-1.0)+100.0)/100.0)
		#convert into a flux per year
		P_minus_E_no_ice *= yearsec
		evap_precip_flag = True
	######################################
	# 3D salinity                        #
	######################################
	S = iris.load_cube(directory+model+'*_so_past1000_r1i1p1_*Omon*.nc')
	#units = psu = g/kg = kg/1000kg. Convert it to kg m-3
	S /= 1.026 # Change this so it uses the actual grid point density
	######################################
	# u velocity                         #
	######################################
	uo = iris.load_cube(directory+model+'*_uo_past1000_r1i1p1_*Omon*.nc')
	######################################
	# v velocity                         #
	######################################
	vo = iris.load_cube(directory+model+'*_vo_past1000_r1i1p1_*Omon*.nc')
	##########################################################################
	# stress on the liquid ocean from overlying atmosphere, sea ice, ice shel#
	##########################################################################
	tauu = iris.load_cube(directory+model+'*_tauu_past1000_r1i1p1_*Amon*.nc') 
	tauv = iris.load_cube(directory+model+'*_tauv_past1000_r1i1p1_*Amon*.nc') 
	tau = iris.analysis.maths.exponentiate(iris.analysis.maths.exponentiate(tauu,2)+iris.analysis.maths.exponentiate(tauv,2),0.5)
	######################################
	# misc                               #
	######################################
	# latitude field
	f = tauu.copy()
	latitude = tauu.coord('latitude').points * (np.pi / 180.0) # in radians
	sin_latitude = np.swapaxes(np.tile(np.sin(latitude),[np.shape(tauu)[0],360,1]),1,2)
	#coriolis parameter
	f.data = 2.0 * 2.0* np.pi / (24.0*60.0*60.0) * sin_latitude
	######################################
	# hight of the mixed layer (mld)     #
	######################################
	#         try: #If model supplied MLD
	#         	hm = iris.load_cube(directory+model+'*_mlotst_past1000_r1i1p1_*Omon*.nc')
	#         else:
	# 			S = iris.load_cube(directory+model+'*_so_past1000_*Omon*.nc') 
	# 			thetao = iris.load_cube(directory+model+'*_thetao_past1000_*Omon*.nc') 
	# 			#S = S[0:2]
	# 			#thetao = thetao[0:2]
	# 			depth_cube = S[0].copy()
	# 			depths = depth_cube.coord('depth').points
	# 			#for memorys sake, do one year at a time..
	# 			depth_cube.data = np.ma.masked_array(np.swapaxes(np.tile(depths,[360,180,1]),0,2))
	# 			hm = S.extract(iris.Constraint(depth = np.min(depth_cube.data)))
	# 			latitude_deg = depth_cube.copy()
	# 			latitude_deg_data = hm.coord('latitude').points
	# 			latitude_deg.data = np.swapaxes(np.tile(latitude_deg_data,[np.shape(S)[1],360,1]),1,2)
	# 			hm.data = mld(S,thetao,depth_cube,latitude_deg)
	hm = iris.load_cube(directory+model+'_my_mld_'+ensemble+'.nc')
	dhm = hm[0:-1].copy()
	dhm.data = hm[1::].data-hm[0:-1].data
	grad_hm = hm.copy()
	tmp = np.gradient(hm.data)
	grad_hm.data = tmp[1]+tmp[2]
	######################################
	# ekman pumping velocity             #
	######################################
	tmp = (tau/(po*f))
	wek = tmp.copy()
	tmp2 = np.gradient(tmp.data)
	wek.data = tmp2[1]+tmp2[2]
	##################################################################################
	# Entrainment velocity (change in mixed later (in one month) plus ekman pumping) #
	##################################################################################
	we = wek[0:-1].copy()
	we.data = dhm.data + wek[0:-1].data
	######################################
	# Ekman velocity                     #
	######################################
	ue = (tau * k) / (po * f * hm)
	#####################################################################
	# the non-Ekman, residual horizontal component of the velocity      #
	#####################################################################
	u = iris.analysis.maths.exponentiate(iris.analysis.maths.exponentiate(uo,2)+iris.analysis.maths.exponentiate(vo,2),0.5)
	u = u.collapsed('depth',iris.analysis.MEAN)
	ue.units ='meter-second^-1'
	u -= ue
	######################################
	# mixed layer salinity               #
	######################################
	depth_cube = S[0].copy()
	depths = depth_cube.coord('depth').points
	depth_cube.data = np.ma.masked_array(np.swapaxes(np.tile(depths,[360,180,1]),0,2))
	thickness_cube = S[0].copy()
	thicknesses = depth_cube.coord('depth').bounds[:,1] - depth_cube.coord('depth').bounds[:,0]
	thickness_cube.data = np.ma.masked_array(np.swapaxes(np.tile(thicknesses,[360,180,1]),0,2))
	#  mask everywhere below the mixed layer and calculate the mean mixed layer salinity #
	s_mixed_layer = S.extract(iris.Constraint(depth = depths[0]))
	s_mixed_layer_data = s_mixed_layer.data.copy()
	for time_index,cube_tmp in enumerate(S.slices(['depth','latitude','longitude'])):
		print time_index
		thickness_cube2 = thickness_cube.copy()
		thickness_cube2_data = thickness_cube2.data
		tmp_data = cube_tmp.data
		for depth in np.arange(np.size(depths)):
			tmp_data[depth,:,:] = np.ma.masked_where(hm[time_index,:,:].data < depths[depth],tmp_data[depth,:,:])
			thickness_cube2_data[depth,:,:] = np.ma.masked_where(hm[time_index,:,:].data < depths[depth],thickness_cube2_data[depth,:,:])
		cube_tmp.data = tmp_data
		cube_tmp *= thickness_cube
		s_mixed_layer_data[time_index,:,:] = cube_tmp.collapsed(['depth'],iris.analysis.SUM).data
		thickness_cube2.data = thickness_cube2_data
		total_thickness_cube = thickness_cube2.collapsed(['depth'],iris.analysis.SUM).data
		s_mixed_layer_data[time_index,:,:] /= total_thickness_cube.data
		#plt.contourf(s_mixed_layer_data[time_index,:,:],21)
		#plt.colorbar()
		#plt.show()
	#####################################################################
	# s_mixed_layer contains the average salinity of the mixed layer    #
	#####################################################################
	s_mixed_layer.data = s_mixed_layer_data
	grad_s_mixed_layer = s_mixed_layer.copy()
	tmp = np.gradient(s_mixed_layer.data)
	grad_s_mixed_layer.data = tmp[1]+tmp[2]
	#####################################################################...
	# freshwater flux into the ocean due to cryospheric (sea ice) melt. We thus estimate Fi by subtracting the (E-P) contribution from the total freshwater flux into the ocean, and applying a mask that sets the contribution to zero when there is no sea ice coverage (thus removing the residual contribution due to runoff at lower latitudes)
	#####################################################################...
	# 	P_minus_E_not_where_ice = P_no_ice - E_no_ice
	# total freshwater flux in to ocean
	if not evap_precip_flag:
		wfo = iris.load_cube(directory+model+'*_wfo_past1000_r1i1p1_*Omon*.nc')
		wfo *= yearsec
		Fi = wfo - P_minus_E_no_ice
		sic.data[np.where(sic.data > 0)] = 1.0 
		Fi *= sic # note that we set this to zero where there is no sea ice
	######################################
	######################################
	### Components of salinity change  ###
	######################################
	######################################
	#E-P driven salinity change (kg salt per m-3 per year?)
	EP_contribution = (P_minus_E_no_ice * s_mixed_layer/1000.0) / hm
	#initially just look at this as fraction of the whole salinity change - how much of it can be explained?
	#Ekman horizontal advective flux
	ekman_hor_adv_contribution = ue * grad_s_mixed_layer
	# non ekman horizontal advective flux
	non_ekman_hor_hor_adv_contribution = u * grad_s_mixed_layer
	# Diffusive contribution
	laplace_s_mixed_layer = s_mixed_layer.copy()     
	laplace_s_mixed_layer.data = scipy.ndimage.filters.laplace(s_mixed_layer.data)
	diffusive_contribution = k * laplace_s_mixed_layer
	# Vertical entrainment
	vertical_entrainment_contribution = (we * grad_s_mixed_layer[0:-1]) / hm[0:-1]
	# Lateral entrainment
	lateral_entrainment_contribution = ((u * grad_hm) * grad_s_mixed_layer) / hm
	# brine/meltwater input from the sea ice formation cycle
	seaice_contribution = s_mixed_layer[1::].copy()
	if not evap_precip_flag:
		seaice_contribution = (((pi * Si) - (po * s_mixed_layer)) * Fi) / (po * hm)
	else:
		seaice_contribution.data = (s_mixed_layer[1::].data - s_mixed_layer[0:-1].data) - EP_contribution[0:-1].data - ekman_hor_adv_contribution[0:-1].data - non_ekman_hor_hor_adv_contribution[0:-1].data + diffusive_contribution[0:-1].data - vertical_entrainment_contribution.data - lateral_entrainment_contribution[0:-1].data
	model_data[model]['mixed_layer_salinity'] = s_mixed_layer
	#NOTE RUNS DSO FAR HAVE BEEN 1e3 too low - had an spurios divide by 1000. Now removed for any future anlaysusl/...
	model_data[model]['EP_contribution'] = EP_contribution
	model_data[model]['ekman_hor_adv_contribution'] = ekman_hor_adv_contribution
	model_data[model]['non_ekman_hor_hor_adv_contribution'] = non_ekman_hor_hor_adv_contribution
	model_data[model]['diffusive_contribution'] = diffusive_contribution
	model_data[model]['vertical_entrainment_contribution'] = vertical_entrainment_contribution
	model_data[model]['lateral_entrainment_contribution'] = lateral_entrainment_contribution
	model_data[model]['seaice_contribution'] = seaice_contribution
	model_data[model]['evap_precip_flag'] = evap_precip_flag
	with open('/data/NAS-ph290/ph290/cmip5/pickles/salinity_budget_plot_3.pickle', 'w') as f:
		pickle.dump([models,model_data], f)


print 'compare results in s.ocean to paper'

#with open('/data/NAS-ph290/ph290/cmip5/pickles/salinity_budget_plot_2.pickle', 'w') as f:
#   pickle.dump([models,model_data], f)


# with open('/data/NAS-ph290/ph290/cmip5/pickles/salinity_budget_plot_3.pickle', 'r') as f:
#    models,model_data = pickle.load(f)

'''
#execfile('/home/ph290/Documents/python_scripts/salinity_budget.py')
'''

'''

model = models[0]

y1 = model_data[model]['mixed_layer_salinity'][0:-1].copy()
y1.data = model_data[model]['mixed_layer_salinity'][1::].data - model_data[model]['mixed_layer_salinity'][0:-1].data


def extract_and_mean(cube):
	west = -24
	east = -13
	south = 65
	north = 67
	cube = cube.intersection(longitude=(west, east))
	cube = cube.intersection(latitude=(south, north))
	try:
		cube.coord('latitude').guess_bounds()
	except:
		print 'cube already has latitude bounds' 
	try:
		cube.coord('longitude').guess_bounds()
	except:
		print 'cube already has longitude bounds'
	grid_areas = iris.analysis.cartography.area_weights(cube)
	return cube.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights=grid_areas)


circulation = model_data[model]['ekman_hor_adv_contribution'][0:-1] + model_data[model]['non_ekman_hor_hor_adv_contribution'][0:-1] + model_data[model]['diffusive_contribution'][0:-1] + model_data[model]['vertical_entrainment_contribution'] + model_data[model]['lateral_entrainment_contribution'][0:-1]


ts1 = extract_and_mean(y1)
ts2 = extract_and_mean(model_data[model]['EP_contribution'])
ts3 = extract_and_mean(circulation)
ts4 = extract_and_mean(model_data[model]['seaice_contribution'])


rm_value = 10

plt.close('all')
plt.plot(rm.running_mean(ts1.data,rm_value),'b')
plt.plot(rm.running_mean(ts2.data/10.0,rm_value),'r')
plt.savefig('/home/ph290/Documents/figures/delete.tmp.png')


plt.plot(rm.running_mean(ts3.data,rm_value),'g')
plt.plot(rm.running_mean(ts4.data,rm_value),'y')
tmp = ts2[0:-1]
tmp.data = (-1.0) * ts2[0:-1].data + ts4.data + ts3.data 
plt.plot(rm.running_mean(tmp.data,rm_value),'k')
plt.savefig('/home/ph290/Documents/figures/delete.tmp.png')


qplt.scatter(ts1*1000.0,tmp)
plt.xlim(-0.2,0.2)
plt.ylim(-0.2,0.2)
plt.show()



qplt.scatter(ts1*1000.0,ts2[0:-1] - ts2[0:-1].collapsed('time',iris.analysis.MEAN))
plt.show()

qplt.scatter(ts1*1000.0,tmp - tmp.collapsed('time',iris.analysis.MEAN))
plt.show()

NOTE that at the moment, teh P-E term seems to be consistantly biased. Unless this is offset by the sea-ice term (check which models have this data), this is confusing. Might be bevause it is multriplied by the wrong saliniyt at presnet?

NOTE also that year to year this is always going to be wrong, 'cos teh salinity change is from one year mean to teh next year mean, and the salinity/freshwater inputs are across just one year - i.e. it is 6 months out of sink

'''

