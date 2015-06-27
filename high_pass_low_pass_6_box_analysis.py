'''
6-box model high/low/band pass analysis
'''

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.signal as signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
import matplotlib.mlab as ml
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
import scipy.stats

markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

styles = markers + [
    r'$\lambda$',
    r'$\bowtie$',
    r'$\circlearrowleft$',
    r'$\clubsuit$',
    r'$\checkmark$']


def rmse(ts1,ts2):
    #ts1 = predicted value
    #ts2 = true value
    arraysize=np.size(ts1)
    diff_sq=np.square(ts1-ts2)
    mse=np.sum(diff_sq)*(1.0/arraysize)
    return np.sqrt(mse)
    



array_size=239

source="/home/ph290/data1/boxmodel_testing_less_boxes_4_annual_matlab_high_low_pass/results/order_runs_processed_in_box_model.txt"
inp = open(source,"r")
count=0
for line in inp:
    count +=1

'''
Read in subpolar gyre co2 flux data from QUMP
'''

run_names_order=str.split(line,' ')
run_names_order=run_names_order[0:-1]

input_year=np.zeros(array_size)
qump_co2_flux=np.zeros(array_size)

model_vars=['stash_30249']


dir_name2=['/home/ph290/data1/qump_out_python/annual_means/','/home/ph290/data1/qump_out_python/annual_means/low_pass_2/','/home/ph290/data1/qump_out_python/annual_means/band_pass_2/']
no_filenames=glob.glob(dir_name2[0]+'*30249.txt')
filenames2=glob.glob(dir_name2[0]+'*'+run_names_order[0]+'*'+model_vars[0]+'.txt')

input2=np.genfromtxt(filenames2[0], delimiter=",")

qump_year=np.zeros(array_size)
qump_data=np.zeros((3,array_size,np.size(no_filenames)))
#first 3 referes to normal, low-pass, band pass

input2=np.genfromtxt(filenames2[0], delimiter=",")
qump_year=input2[:,0]

for k in np.arange(np.size(dir_name2)):
	for i in range(np.size(no_filenames)):
		filenames2=glob.glob(dir_name2[k]+'*'+run_names_order[i]+'*'+model_vars[0]+'.txt')
		input2=np.genfromtxt(filenames2[0], delimiter=",")
		if input2[:,1].size == array_size:
			qump_data[k,:,i]=input2[:,1]
			#should this be column 1???? Is this box 1?

#testing...
for k in np.arange(np.size(dir_name2)):
    for i in range(np.size(no_filenames)):
        filenames2=glob.glob(dir_name2[k]+'*'+run_names_order[i]+'*'+model_vars[0]+'.txt')
        input2=np.genfromtxt(filenames2[0], delimiter=",")
	if input2[:,1].size == array_size:
            qump_data[k,:,i]=qump_data[k,:,i]-np.mean(qump_data[k,:,i])


    
		
'''
Read in subpolar gyre co2 flux data from box model
simulations with various smoothings...
'''

dir_name='/home/ph290/box_modelling/boxmodel_6_box_filtered_vars/results/'
filenames=glob.glob(dir_name+'box_model*.csv')
input1=np.genfromtxt(filenames[0], delimiter=",")


filenamesc=glob.glob(dir_name+'box_model_qump_results_1_*.csv')
filenamesb=glob.glob(dir_name+'box_model_qump_results_?_1.csv')
file_order=np.empty(np.size(filenames))
file_order2=np.empty(np.size(filenames))

box_co2_flux=np.zeros((array_size,np.size(no_filenames),np.size(filenamesc),np.size(filenamesb)))
box_years=input1[:,0]

for i in range(np.size(filenames)):
    tmp=filenames[i].split('_')
    tmp2=tmp[9].split('.')
    file_order[i]=np.int(tmp2[0])
    tmp3=tmp[10].split('.')
    file_order2[i]=np.int(tmp3[0])

for i in np.arange(np.size(filenamesb)):
    for j in np.arange(np.size(filenamesc)):
        loc=np.where((file_order == i+1) & (file_order2 == j+1))
        #print 'reading box model file '+str(i)
        filenames[loc[0]]
        print filenames[loc[0]]
        input1=np.genfromtxt(filenames[loc[0]], delimiter=",")
        box_co2_flux[:,:,j,i]=input1[:,1:np.size(no_filenames)+1]
    

#tetsing....
shape_tmp=box_co2_flux.shape
for i in np.arange(shape_tmp[1]):
    for j in np.arange(shape_tmp[2]):
        for k in np.arange(shape_tmp[3]):
            box_co2_flux[:,i,j,k]=box_co2_flux[:,i,j,k]-np.mean(box_co2_flux[:,i,j,k])

'''
Now compare the two above sets of data to get a handle
on what drives the low and high frequency variability
'''

########################
# low pass # band pass #
########################

#overplot on each of the above the box model result with either low/band pass input variables.

plot_names=['1 no smoothing','2 salt smoothed (all boxes - yes to start with) (band pass)','3 temp smoothed (band pass)','4 Alk smoothed (band)','5 moc smoothed (band pass)','6 atm co2 smoothed (band pass)','7 alk amd co2 (band pass)','8 all band passed','9 all band passed, salt const','10 all band passed, temp const','11 all band passed, alk const','12 all band passed, moc const','13 all band passed, temp and alk const','14 alk, co2 and T (band pass)','15 All band passed','16 no smoothing, salt const','17 no smoothing, temp const','18 no smoothing, alk const','19 no smoothing, moc const','20 no smoothing, atm co2 const','21 no smoothing temp and alk const']

for i in range(1):
	fig = plt.figure(figsize=(5*3.13,3*3.13))
	ax1 = fig.add_subplot(3,1,1)
	lns=plt.plot(qump_year,qump_data[0,:,i],c='b',linewidth=2,label='QUMP flux')
	for k in np.append(np.arange(0,6),[11]):
		#ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,marker=styles[k],markersize=4,label=plot_names[k])
		ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,label=plot_names[k])
		lns += ln_tmp
		#this should plot out the variously filtered (at input to box model) box model results,
		#for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux

	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs).draw_frame(False)

	ax2 = fig.add_subplot(3,1,2)
	lns=plt.plot(qump_year,qump_data[0,:,i],c='b',linewidth=2,label='QUMP flux')
	for k in np.arange(7):
		ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,marker=styles[k],markersize=4,label=plot_names[k])
		lns += ln_tmp
		#this should plot out the variously filtered (at input to box model) box model results,
		#for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux
	
	plt.plot(qump_year,qump_data[0,:,i])

	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs).draw_frame(False)

	ax3 = fig.add_subplot(3,1,3)
	lns=plt.plot(qump_year,qump_data[2,:,i]-np.mean(qump_data[2,:,i]),c='b',linewidth=2,label='QUMP flux')
	for k in np.arange(7,13):
		ln_tmp=plt.plot(qump_year,((box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0)-np.mean((box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0),marker=styles[k],markersize=4,label=plot_names[k])
		lns += ln_tmp
		#this should plot out the variously filtered (at input to box model) box model results,
		#for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux
	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs).draw_frame(False)

	plt.tight_layout()
	plt.show()

colours=['b','g','r','c','k']
sym_size_var=0.1
line_width_var=0.1

fig = plt.figure(figsize=(5*3.13,3*3.13))
for i,k in enumerate(np.append(np.arange(7),[13])):
    ax1 = fig.add_subplot(2,4,i+1)
    for j in np.arange(np.size(filenamesb)):
        ax1.scatter(qump_data[0,:,:],(box_co2_flux[:,:,k,j]/1.12e13)*1.0e15/12.0, s=sym_size_var, facecolor='1.0',color=colours[j], lw = line_width_var)
        plt.plot([-10,10],[-10,10])
        plt.xlim(-5,10)
        plt.ylim(-5,10)
        plt.title(plot_names[k])
        plt.xlabel('ESM data')
        plt.ylabel('Box model data')

plt.tight_layout()
plt.show()
	
#details needed for band pass filtering data so that we are comparing just the variability rather than the longer then change
N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
low_cutoff=20.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
high_cutoff=1.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
middle_cuttoff_low=1.0
middle_cuttoff_high=5.0

Wn_low=timestep_between_values/low_cutoff
Wn_high=timestep_between_values/high_cutoff
Wn_mid_low=timestep_between_values/middle_cuttoff_low
Wn_mid_high=timestep_between_values/middle_cuttoff_high

#design butterworth filters - or if want can replace butter with bessel
b, a = scipy.signal.butter(N, Wn_low, btype='low')
b1, a1 = scipy.signal.butter(N, Wn_mid_low, btype='low')
b2, a2 = scipy.signal.butter(N, Wn_mid_high, btype='high')

#plot_names=['1 no smoothing','2 salt smoothed (all boxes - yes to start with) (band pass)','3 temp smoothed (band pass)','4 Alk smoothed (band)','5 moc smoothed (band pass)','6 atm co2 smoothed (band pass)','7 alk amd co2 (band pass)','8 all band passed','9 all band passed, salt const','10 all band passed, temp const','11 all band passed, alk const','12 all band passed, moc const','13 all band passed, temp and alk const','14 alk, co2 and T (band pass)','15 All band passed','16 no smoothing, salt const','17 no smoothing, temp const','18 no smoothing, alk const','19 no smoothing, moc const','20 no smoothing, atm co2 const','21 no smoothing temp and alk const']

fig = plt.figure(figsize=(5*3.13,3*3.13))
for i,k in enumerate(np.arange(14,21)):  
    ax1 = fig.add_subplot(2,4,i+1)
    for j in np.arange(np.size(filenamesb)):
        tmp_shape=box_co2_flux[:,:,k,j].shape
        for l in range(tmp_shape[1]):
            dat=((box_co2_flux[:,l,k,j]/1.12e13)*1.0e15/12.0)
            dat_tmp=scipy.signal.filtfilt(b1, a1, dat)
            dat_band_pass=scipy.signal.filtfilt(b2, a2, dat_tmp)
            #mdat=np.ma.masked_array(dat_band_pass,np.isnan(dat_band_pass))
            dat2=qump_data[2,:,l]
            dat2_tmp=scipy.signal.filtfilt(b1, a1, dat2)
            dat2_band_pass=scipy.signal.filtfilt(b2, a2, dat2_tmp)
            #mdat2=np.ma.masked_array(dat2_band_pass,np.isnan(dat2_band_pass))
            ax1.scatter(dat2_band_pass,dat_band_pass, s=sym_size_var, facecolor='1.0',color=colours[j], lw = line_width_var)
            plt.plot([-10,10],[-10,10])
            plt.xlim(-2,2)
            plt.ylim(-2,2)
            plt.title(plot_names[k])
            plt.xlabel('ESM data')
            plt.ylabel('Box model data')

plt.tight_layout()
plt.show()

'''
calculating ensemble member RMSEs or should I do R2s?
'''

r2=np.zeros(tmp_shape[1])
r2.fill(np.nan)

for i in np.arange(tmp_shape[1]):
    qump=qump_data[0,:,i]
    box=(box_co2_flux[:,i,0,0]/1.12e13)*1.0e15/12.0
    tmp=np.where(np.logical_not(np.isnan(box)))
    qump=qump[tmp]
    box=box[tmp]
    if qump.size <> 0:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(qump,box)
        r2[i]=r_value**2
  
r2_sorted=np.sort(r2)
r2_sorted=r2_sorted[np.logical_not(np.isnan(r2_sorted))]

no=5
worst=np.zeros(no)
best=np.zeros(no)

for i in np.arange(np.size(best)):
    worst[i]=np.where(r2 == r2_sorted[i])[0]
    best[i]=np.where(r2 == r2_sorted[-1*(i+1)])[0]

'''
'''


lns=[]
ln_tmp=[]


fig = plt.figure(figsize=(30,20))

for i in np.arange(np.size(best)):
	ax1 = fig.add_subplot(2,no,i+1)
	lns=plt.plot(qump_year,qump_data[0,:,best[i]],c='b',linewidth=2,label='QUMP flux')
	for k in np.array([0,3,5]):
		ln_tmp=plt.plot(qump_year,(box_co2_flux[:,best[i],k,0]/1.12e13)*1.0e15/12.0,label=plot_names[k])
		plt.title('Best: '+np.str(i+1))
		lns += ln_tmp

for i in np.arange(np.size(best)):
	ax1 = fig.add_subplot(2,no,i+1+np.size(best))
	lns=plt.plot(qump_year,qump_data[0,:,worst[i]],c='b',linewidth=2,label='QUMP flux')
	for k in np.array([0,3,5]):
		ln_tmp=plt.plot(qump_year,(box_co2_flux[:,worst[i],k,0]/1.12e13)*1.0e15/12.0,label=plot_names[k])
		plt.title('Worst: '+np.str(i+1))
		lns += ln_tmp


labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=0).draw_frame(False)


#plt.tight_layout()
plt.show()


'''
multmodel mean
'''
tmp_shape=box_co2_flux.shape
box_co2_flux_mean=np.zeros([tmp_shape[0],tmp_shape[2],tmp_shape[3]])

for i1 in np.arange(tmp_shape[3]):
    for i2 in np.arange(tmp_shape[2]):
        for i3 in np.arange(tmp_shape[0]):
            tmp=box_co2_flux[i3,:,i2,i1]
	    tmp=tmp[np.logical_not(np.isnan(tmp))]
            box_co2_flux_mean[i3,i2,i1]=np.mean(tmp)


qump_co2_flux_mean=np.zeros(tmp_shape[0])

for i in np.arange(tmp_shape[0]):
    tmp=qump_data[0,i,:]
    tmp=tmp[np.logical_not(np.isnan(tmp))]
    qump_co2_flux_mean[i]=np.mean(tmp)

lns=[]
ln_tmp=[]


fig = plt.figure(figsize=(30,20))

start=5

lns=plt.plot(qump_year,qump_co2_flux_mean,c='b',linewidth=2,label='QUMP flux')
#or k in np.array([0,7,8,9,10,19,20,21]):
for k in np.array([0,3,5]):
	ln_tmp=plt.plot(qump_year,(box_co2_flux_mean[:,k,0]/1.12e13)*1.0e15/12.0,label=plot_names[k])
	lns += ln_tmp
	#this should plot out the variously filtered (at input to box model) box model results,
	#for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux

labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=0).draw_frame(False)


plt.tight_layout()
plt.show()

'''
'''

print 'this is using the simulation with everything varying as normal, but one variabile held constant at its mean value from the 1st 20 years'

lns=[]
ln_tmp=[]

tmp_shape=box_co2_flux.shape
l=0

fig = plt.figure(figsize=(30,20))

start=5
for i in (np.arange(6)+start):

        dat2=qump_data[0,:,i]
        dat2_tmp=scipy.signal.filtfilt(b1, a1, dat2)
        dat2_band_pass=scipy.signal.filtfilt(b2, a2, dat2_tmp)
        mdat2=np.ma.masked_array(dat2_band_pass,np.isnan(dat2_band_pass))

	ax1 = fig.add_subplot(2,3,i-start)

	lns=plt.plot(qump_year,mdat2,c='b',linewidth=2,label='QUMP flux')
	for k in [0]:
#np.append([0],np.arange(15,21)):

            

            dat=((box_co2_flux[:,i,k,l]/1.12e13)*1.0e15/12.0)
            dat_tmp=scipy.signal.filtfilt(b1, a1, dat)
            dat_band_pass=scipy.signal.filtfilt(b2, a2, dat_tmp)
            mdat=np.ma.masked_array(dat_band_pass,np.isnan(dat_band_pass))



            #ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,marker=styles[k],markersize=4,label=plot_names[k])
            ln_tmp=plt.plot(qump_year,mdat,label=plot_names[k])
            lns += ln_tmp
            #this should plot out the variously filtered (at input to box model) box model results,
            #for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux

labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=0).draw_frame(False)


plt.tight_layout()
plt.show()


'''
'''

print 'this version of the plot uses teh analsysis done with proper band-pass filtered data all the way...'

lns=[]
ln_tmp=[]

tmp_shape=box_co2_flux.shape
l=0

fig = plt.figure(figsize=(30,20))

start=0
for i in (np.arange(6)+start):

        dat2=qump_data[2,:,i]
        #dat2_tmp=scipy.signal.filtfilt(b1, a1, dat2)
        #dat2_band_pass=scipy.signal.filtfilt(b2, a2, dat2_tmp)
        #mdat2=np.ma.masked_array(dat2_band_pass,np.isnan(dat2_band_pass))

	ax1 = fig.add_subplot(2,3,i-start)

	lns=plt.plot(qump_year,dat2,c='b',linewidth=2,label='QUMP flux')
	for k in np.append(np.arange(7,13),[14]):

            

            dat=((box_co2_flux[:,i,k,l]/1.12e13)*1.0e15/12.0)
            #dat_tmp=scipy.signal.filtfilt(b1, a1, dat)
            #dat_band_pass=scipy.signal.filtfilt(b2, a2, dat_tmp)
            #mdat=np.ma.masked_array(dat_band_pass,np.isnan(dat_band_pass))



            #ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,marker=styles[k],markersize=4,label=plot_names[k])
            ln_tmp=plt.plot(qump_year,dat,label=plot_names[k])
            lns += ln_tmp
            #this should plot out the variously filtered (at input to box model) box model results,
            #for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux

labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=0).draw_frame(False)
plt.show()

'''
best and worst
'''



r2=np.zeros(tmp_shape[1])
r2.fill(np.nan)

for i in np.arange(tmp_shape[1]):
    qump=qump_data[2,:,i]
    box=(box_co2_flux[:,i,7,0]/1.12e13)*1.0e15/12.0
    tmp=np.where(np.logical_not(np.isnan(box)))
    qump=qump[tmp]
    box=box[tmp]
    if qump.size <> 0:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(qump,box)
        r2[i]=r_value**2
  
r2_sorted=np.sort(r2)
r2_sorted=r2_sorted[np.logical_not(np.isnan(r2_sorted))]


no=1
worst=np.zeros(no)
best=np.zeros(no)

for i in np.arange(np.size(best)):
    worst[i]=np.where(r2 == r2_sorted[i])[0]
    best[i]=np.where(r2 == r2_sorted[-1*(i+1)])[0]

lns=[]
ln_tmp=[]


fig = plt.figure(figsize=(30,20))

for i in np.arange(np.size(best)):
	ax1 = fig.add_subplot(2,no,i+1)
	lns=plt.plot(qump_year,qump_data[2,:,best[i]],c='b',linewidth=2,label='QUMP flux')
	for k in [7,9,10,12]:
		ln_tmp=plt.plot(qump_year,(box_co2_flux[:,best[i],k,0]/1.12e13)*1.0e15/12.0,label=plot_names[k])
		plt.title('Best: '+np.str(i+1))
		lns += ln_tmp

for i in np.arange(np.size(best)):
	ax1 = fig.add_subplot(2,no,i+1+np.size(best))
	lns=plt.plot(qump_year,qump_data[2,:,worst[i]],c='b',linewidth=2,label='QUMP flux')
	for k in [7,9,10,12]:
		ln_tmp=plt.plot(qump_year,(box_co2_flux[:,worst[i],k,0]/1.12e13)*1.0e15/12.0,label=plot_names[k])
		plt.title('Worst: '+np.str(i+1))
		lns += ln_tmp


labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=0).draw_frame(False)
plt.show()

