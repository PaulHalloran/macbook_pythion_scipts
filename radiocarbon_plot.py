import numpy as np
import matplotlib.pyplot as plt

file = '/Users/ph290/SparkleShare/halloran/data/c14.txt'

file2 = '/Users/ph290/SparkleShare/halloran/data/nydal.asc'

data = np.genfromtxt(file,skip_header = 16)

f = open(file2,'r')
datatmp = f.read()
f.close()
datatmp = datatmp.split('\n')
dates = []
data2 = []
for i,line in enumerate(datatmp):
	if (i > 16) & (i < 1635):
		date = line.split('-')[1][-6:]
		yr = float('19'+date[0:2])
		mn = float(date[2:4])
		dy = float(date[4:6])
		date = yr + mn/12.0 + dy/30.0
		dates.append(date)
		data2.append(line.split(' ')[3])		

dates = np.array(dates,float)
data2 = np.array(data2,float)


plt.close('all')
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax1.scatter(data[:,0],data[:,1])
ax1.set_xlim([1800,1960])
ax1.set_ylabel('delta $^{14}$C')
ax1.set_xlabel('calendar year')
ax2 = fig.add_subplot(122)
ax2.scatter(dates,data2)
ax2.set_xlim([1960,2000])


plt.savefig('/Users/ph290/SparkleShare/halloran sync/documents/proposals/past_air_sea_flux/figures/atm_d14c.pdf')


