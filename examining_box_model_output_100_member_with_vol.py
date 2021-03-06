import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.signal as signal
from scipy.signal import kaiserord, lfilter, firwin, freqz

def rmse(ts1,ts2):
    #ts1 = predicted value
    #ts2 = true value
    arraysize=np.size(ts1)
    diff_sq=np.square(ts1-ts2)
    mse=np.sum(diff_sq)*(1.0/arraysize)
    return np.sqrt(mse)

array_size=2880

source=r"/net/project/obgc/boxmodel_testing_less_boxes_4_monthly_inclusing_vol_params/results/order_runs_processed_in_box_model.txt"
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


dir2='/net/project/obgc/qump_out_python/'
no_filenames=glob.glob(dir2+'*30249.txt')
filenames2=glob.glob(dir2+'*'+run_names_order[0]+'*30249.txt')

qump_year=np.zeros(array_size)
qump_co2_flux=np.zeros((array_size,np.size(no_filenames)))

input2=np.genfromtxt(filenames2[0], delimiter=",")
qump_year=input2[:,0]

for i in range(np.size(no_filenames)):
    filenames2=glob.glob(dir2+'*'+run_names_order[i]+'*30249.txt')
    input2=np.genfromtxt(filenames2[0], delimiter=",")
    if input2[:,1].size == array_size:
        qump_co2_flux[:,i]=input2[:,1]

'''
Read in data from box model
'''

dir='/net/project/obgc/boxmodel_testing_less_boxes_4_monthly_inclusing_vol_params/results/'
filenames=glob.glob(dir+'box_model*.csv')
input1=np.genfromtxt(filenames[0], delimiter=",")
box_co2_flux=np.zeros((array_size,np.size(no_filenames),np.size(filenames)))
box_years=input1[:,0]

for i in range(np.size(filenames)):
    #print 'reading box model file '+str(i)
    input1=np.genfromtxt(filenames[i], delimiter=",")
    box_co2_flux[:,:,i]=input1[:,1:np.size(no_filenames)+1]

'''
Analysis
'''


'''
unsmoothed
'''


smoothing_len=12.0
#plt.plot(box_years[smoothing_len/2.0:-smoothing_len/2.0+1],matplotlib.mlab.movavg((box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0,smoothing_len),'k')
#plt.plot(qump_year[smoothing_len/2.0:-smoothing_len/2.0+1],matplotlib.mlab.movavg(qump_co2_flux[:,exp_counter],smoothing_len),'r')
#plt.show()

#ts1=matplotlib.mlab.movavg((box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0,smoothing_len)
#ts2=matplotlib.mlab.movavg(qump_co2_flux[:,exp_counter],smoothing_len)
#rmse(ts1,ts2)

'''
unsmoothed
'''

#plt.plot(box_years,(box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0,'k')
#plt.plot(qump_year,qump_co2_flux[:,exp_counter],'r')
#plt.show()

#ts1=box_co2_flux[:,exp_counter,param_set_counter]
#ts2=qump_co2_flux[:,exp_counter]
#rmse(ts1,ts2)


'''
once run, process, and pull out lowest RMSEs
'''


'''
and low/high-pass filtering
'''
N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0/12.0 #years valkue should be '1.0/12.0'
low_cutoff=10.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
high_cutoff=1.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
middle_cuttoff_low=1.0
middle_cuttoff_high=5.0

Wn_low=timestep_between_values/low_cutoff
Wn_high=timestep_between_values/high_cutoff
Wn_mid_low=timestep_between_values/middle_cuttoff_low
Wn_mid_high=timestep_between_values/middle_cuttoff_high

param_set_counter=328

def analysis():
    for exp_counter in range(np.size(no_filenames)):
        print exp_counter
        #
        #design a butterworth filter - or of want can replace butter with bessel
        # low-pass
        b, a = scipy.signal.butter(N, Wn_low, btype='low')
        qump_low_pass = scipy.signal.filtfilt(b, a, qump_co2_flux[:,exp_counter])
        box_low_pass = scipy.signal.filtfilt(b, a,(box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0)
        # band-pass
        b1, a1 = scipy.signal.butter(N, Wn_mid_low, btype='low')
        b2, a2 = scipy.signal.butter(N, Wn_mid_high, btype='high')
        qump_middle_tmp=scipy.signal.filtfilt(b1, a1, qump_co2_flux[:,exp_counter])
        qump_middle=scipy.signal.filtfilt(b2, a2, qump_middle_tmp)
        box_middle_tmp=scipy.signal.filtfilt(b1, a1, (box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0)
        box_middle=scipy.signal.filtfilt(b2, a2, box_middle_tmp)
        # high-pass
        b, a = scipy.signal.butter(N, Wn_high, btype='high')
        qump_high_pass = scipy.signal.filtfilt(b, a, qump_co2_flux[:,exp_counter])
        box_high_pass = scipy.signal.filtfilt(b, a,(box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0)
        # plotting
        f, axarr = plt.subplots(5, sharex=True)
        axarr[0].plot(qump_year, qump_co2_flux[:,exp_counter],'k')
        axarr[0].plot(qump_year, (box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0,'r')
        axarr[0].set_title('unfiltered monthly data')
        smoothing_len=12.0
        axarr[1].plot(box_years[smoothing_len/2.0:-smoothing_len/2.0+1],matplotlib.mlab.movavg((box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0,smoothing_len),'r')
        axarr[1].plot(qump_year[smoothing_len/2.0:-smoothing_len/2.0+1],matplotlib.mlab.movavg(qump_co2_flux[:,exp_counter],smoothing_len),'k')
        axarr[1].set_title('moving avg. smoothed (12 month window)')
        axarr[2].plot(qump_year[round((12*low_cutoff)/2.0):-round((12*low_cutoff)/2.0)], qump_low_pass[round((12*low_cutoff)/2.0):-round((12*low_cutoff)/2.0)],'k')
        axarr[2].plot(qump_year[round((12*low_cutoff)/2.0):-round((12*low_cutoff)/2.0)], box_low_pass[round((12*low_cutoff)/2.0):-round((12*low_cutoff)/2.0)],'r')
        axarr[2].set_title('low-pass > '+str(low_cutoff)+'yrs')
        axarr[3].plot(qump_year[round((12*middle_cuttoff_low)/2.0):-round((12*middle_cuttoff_low)/2.0)], qump_middle[round((12*middle_cuttoff_low)/2.0):-round((12*middle_cuttoff_low)/2.0)],'k')
        axarr[3].plot(qump_year[round((12*middle_cuttoff_low)/2.0):-round((12*middle_cuttoff_low)/2.0)], box_middle[round((12*middle_cuttoff_low)/2.0):-round((12*middle_cuttoff_low)/2.0)],'r')
        axarr[3].set_title('bandpass '+str(middle_cuttoff_high)+' < filter < '+str(middle_cuttoff_low))
        axarr[4].plot(qump_year[round((12*high_cutoff)/2.0):-round((12*high_cutoff)/2.0)], qump_high_pass[round((12*high_cutoff)/2.0):-round((12*high_cutoff)/2.0)],'k')
        axarr[4].plot(qump_year[round((12*high_cutoff)/2.0):-round((12*high_cutoff)/2.0)], box_high_pass[round((12*high_cutoff)/2.0):-round((12*high_cutoff)/2.0)],'r')
        axarr[4].set_title('high-pass < '+str(high_cutoff)+'yrs')
        plt.show()
        continue


# print 'low-pass rmse= '+str(rmse(box_low_pass,qump_low_pass))
# print 'band-pass rmse= '+str(rmse(box_middle,qump_middle))
# print 'high-pass rmse= '+str(rmse(box_high_pass,qump_high_pass))


# mean_rmse = np.zeros(np.size(filenames))

# for param_set_counter in range(np.size(filenames)):
#     rmse_tmp=np.empty(np.size(filenames))
#     rmse_tmp.fill(np.nan)
#     for exp_counter in range(np.size(no_filenames)):
#         rmse_tmp[exp_counter]=(rmse((box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0,qump_co2_flux[:,exp_counter]))
#     rmse_tmp_masked = np.ma.masked_array(rmse_tmp,np.isnan(rmse_tmp))
#     mean_rmse[param_set_counter]=np.mean(rmse_tmp_masked)

# min_rmse=np.min(mean_rmse)
# index=np.where(mean_rmse == min_rmse)
# print index




'''
works out the eman RMSE from each parameter set across each exteriment for both the band-pass (~interannual variability) and low-pass (centenial variability), meaned together
'''

mean_rmse = np.zeros(np.size(filenames))

b, a = scipy.signal.butter(N, Wn_low, btype='low')
b1, a1 = scipy.signal.butter(N, Wn_mid_low, btype='low')
b2, a2 = scipy.signal.butter(N, Wn_mid_high, btype='high')

for param_set_counter in range(np.size(filenames)):
    rmse_tmp=np.empty(np.size(no_filenames))
    rmse_tmp.fill(np.nan)
    rmse_tmp2=np.empty(np.size(no_filenames))
    rmse_tmp2.fill(np.nan)
    #
    rmse_tmp3=np.empty(np.size(filenames))
    rmse_tmp3.fill(np.nan)
    for exp_counter in range(np.size(no_filenames)):
        qump_low_pass = scipy.signal.filtfilt(b, a, qump_co2_flux[:,exp_counter])
        box_low_pass = scipy.signal.filtfilt(b, a,(box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0)
        rmse_tmp[exp_counter]=rmse(box_low_pass[round((12*low_cutoff)/2.0):-round((12*low_cutoff)/2.0)],qump_low_pass[round((12*low_cutoff)/2.0):-round((12*low_cutoff)/2.0)])
        #qump_middle_tmp=scipy.signal.filtfilt(b1, a1, qump_co2_flux[:,exp_counter])
        #qump_middle=scipy.signal.filtfilt(b2, a2, qump_middle_tmp)
        #box_middle_tmp=scipy.signal.filtfilt(b1, a1, (box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0)
        #box_middle=scipy.signal.filtfilt(b2, a2, box_middle_tmp)
        #rmse_tmp2[exp_counter]=rmse(box_middle[round((12*middle_cuttoff_low)/2.0):-round((12*middle_cuttoff_low)/2.0)],qump_middle[round((12*middle_cuttoff_low)/2.0):-round((12*middle_cuttoff_low)/2.0)])
        # using this to add an extra contrain on variability towards the end of the century - prob remove.
        #rmse_tmp3[exp_counter]=rmse(box_middle[12.0*150:-round((12*middle_cuttoff_low)/2.0)],qump_middle[12.0*150:-round((12*middle_cuttoff_low)/2.0)])
    rmse_tmp_masked = np.ma.masked_array(rmse_tmp,np.isnan(rmse_tmp))
    #rmse_tmp2_masked = np.ma.masked_array(rmse_tmp2,np.isnan(rmse_tmp2))
    #
    #rmse_tmp3_masked = np.ma.masked_array(rmse_tmp3,np.isnan(rmse_tmp3)) 
    mean_rmse[param_set_counter]=np.mean(rmse_tmp_masked)
                                         #+rmse_tmp2_masked+rmse_tmp3_masked)
    

min_rmse=np.min(mean_rmse)
mean_rmse_sorted=np.sort(mean_rmse)
print np.where(mean_rmse == mean_rmse_sorted[0])
print np.where(mean_rmse == mean_rmse_sorted[1])
print np.where(mean_rmse == mean_rmse_sorted[2])


#then try with filtering - mean of low and high-pass filters...

x=np.where(mean_rmse == mean_rmse_sorted[0])
param_set_counter=x[0][0]

analysis()



smoothing_len=12
exp_counter=20
f, axarr = plt.subplots(1, sharex=True)
for param_set_counter in range(np.size(filenames)):
    axarr.plot(box_years[smoothing_len/2.0:-smoothing_len/2.0+1],matplotlib.mlab.movavg((box_co2_flux[:,exp_counter,param_set_counter]/1.12e13)*1.0e15/12.0,smoothing_len))
    axarr.plot(qump_year[smoothing_len/2.0:-smoothing_len/2.0+1],matplotlib.mlab.movavg(qump_co2_flux[:,exp_counter],smoothing_len),'k')

plt.show()
  
