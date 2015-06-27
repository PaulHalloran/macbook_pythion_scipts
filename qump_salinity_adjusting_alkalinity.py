import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
import scipy

'''
Read in QUMP data
'''

array_size=239

source="/home/ph290/data1/boxmodel_testing_less_boxes_4_annual_matlab_high_low_pass/results/order_runs_processed_in_box_model.txt"
inp = open(source,"r")
count=0
for line in inp:
    count +=1


run_names_order=str.split(line,' ')
run_names_order=run_names_order[0:-1]

input_year=np.zeros(array_size)
qump_co2_flux=np.zeros(array_size)

model_vars=['stash_101','stash_102','stash_103','stash_104','stash_200','stash_30249','moc_stm_fun']


dir_name2='/home/ph290/data1/qump_out_python/annual_means/'
no_filenames=glob.glob(dir_name2+'*30249.txt')
filenames2=glob.glob(dir_name2+'*'+run_names_order[0]+'*'+model_vars[0]+'.txt')

input2=np.genfromtxt(filenames2[0], delimiter=",")
no_time_series=input2[0,:].size-1

qump_year=np.zeros(array_size)
qump_data=np.zeros((np.size(model_vars),no_time_series,array_size,np.size(no_filenames)))
#variable,box,years,ensemble no.

input2=np.genfromtxt(filenames2[0], delimiter=",")
qump_year=input2[:,0]

for k,model_var_cnt in enumerate(model_vars):
	for i in range(np.size(no_filenames)):
		filenames2=glob.glob(dir_name2+'*'+run_names_order[i]+'*'+model_var_cnt+'.txt')
		input2=np.genfromtxt(filenames2[0], delimiter=",")
		no_time_series=input2[0,:].size-1
		if input2[:,1].size == array_size:
			for j in range(no_time_series):
				qump_data[k,j,:,i]=input2[:,j+1]
                                #variable,box no. (spg is 0),year,ens. member

qump_data2=np.copy(qump_data)
shape_tmp=qump_data.shape
model_var_cnt=model_vars[3]
for i in range(np.size(no_filenames)):
    filenames2=glob.glob(dir_name2+'*'+run_names_order[i]+'*'+model_var_cnt+'.txt')
    input2=np.genfromtxt(filenames2[0], delimiter=",")
    output2=np.copy(input2)
    for k in range(shape_tmp[2]):
        for j in range(shape_tmp[1]):
            tmp=qump_data[1,j,:,i]
            tmp=tmp[np.logical_not(np.isnan(tmp))]
            output2[k,j+1]=(qump_data[3,j,k,i]/((qump_data[1,j,k,i]*1000.0)+35.0))*(tmp[0]*1000.0)+35.0)
            #qump_data2[3,j,k,i]=(qump_data[3,j,k,i]/((qump_data[1,j,k,i]*1000.0)+35.0))*(np.mean((tmp[0:19]*1000.0)+35.0))
    filename_tmp=filenames2[0].split('/')
    filenames3='/'.join([filename_tmp[0],filename_tmp[1],filename_tmp[2],filename_tmp[3],filename_tmp[4],filename_tmp[5]+'_sal_adjusted_alk',filename_tmp[6]])
    np.savetxt(filenames3,output2, delimiter=',')

print 'output to '+filenames3


fig=plt.figure()
ax1=fig.add_subplot(111)
ax1.plot(qump_data[3,0,:,25])
ax1.plot(output2[:,1],'r')
ax2=ax1.twinx()
ax2.plot(qump_data[1,0,:,25])

plt.show()

