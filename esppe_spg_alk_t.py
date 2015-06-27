import numpy as np
import glob
import matplotlib.pyplot as plt 

directory = '/home/ph290/data1/qump_out_python/annual_means/'

t_files = glob.glob(directory+'*101.txt')
alk_files = glob.glob(directory+'*104.txt')

t_yr = []
t_data = []
alk_yr = []
alk_data = []

for file in t_files:
    data = np.genfromtxt(file, delimiter=",")
    yr = data[:,0]
    spg = data[:,1]
    t_yr.append(yr)
    t_data.append(spg)

for file in alk_files:
    data = np.genfromtxt(file, delimiter=",")
    yr = data[:,0]
    spg = data[:,1]
    alk_yr.append(yr)
    alk_data.append(spg)

fig = plt.figure()
fig.add_subplot(121)
for i,data in enumerate(t_data):
    plt.plot(t_yr[i],data - np.mean(data[0:20]))

plt.xlabel('year')
plt.ylabel('Temperature anomaly (K)')

fig.add_subplot(122)
for i,data in enumerate(alk_data):
    plt.plot(t_yr[i],data - np.mean(data[0:20]))

plt.xlabel('year')
plt.ylabel('Alkalinty anomaly ($\mu$mol kg$^{-1}$)')

# plt.tight_layout()
plt.show()

