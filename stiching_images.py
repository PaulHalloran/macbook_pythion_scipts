import cv2
import numpy as np
from matplotlib import pyplot as plt


directory = '/home/ph290/Downloads/iiia13entrada/iiia13entrada-cyl-pano01/'

img = cv2.imread(directory+'cyl_image00.png')

plt.close('all')
plt.imshow(img)
plt.show(block = False)