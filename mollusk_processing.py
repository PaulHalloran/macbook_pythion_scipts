import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filter
import scipy.stats

'''
look at http://scikit-image.org/docs/dev/auto_examples/
'''

file = '/Users/ph290/Downloads/treering.jpg' 

im = np.array(Image.open(file))[:,:,0]

edges = filter.canny(im, sigma=1)

y_section = 650
data = edges[:,y_section]
loc = np.where(data == 1)[0]
widths = []
for i in np.arange(np.size(loc)-1):
	widths = np.append(widths,loc[i+1]-loc[i])

widths2=[]
data = edges[:,y_section-4]
loc = np.where(data == 1)[0]
for i in np.arange(np.size(loc)-1):
        widths2 = np.append(widths2,loc[i+1]-loc[i])

plt.contourf(im,51,cmap=plt.cm.gist_gray)
#plt.contour(edges)
plt.plot([0,1000],[y_section,y_section])
plt.show()

plt.contourf(im,51,cmap=plt.cm.gist_gray)
plt.contour(edges,color = 'r')
plt.plot([0,1000],[y_section,y_section])
plt.show()

roll_val=[]
corr = 0.0
for i in np.arange(20):
	corr_tmp = scipy.stats.pearsonr(widths,np.roll(widths2[np.arange(np.size(widths))],i))
	if corr_tmp >= corr:
		corr = corr_tmp
		roll_val = i


plt.plot(widths)
plt.plot(np.roll(widths2,roll_val))
plt.show()




