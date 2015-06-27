import numpy as np
import matplotlib.pyplot as plt
#loading up modules required to to do
#numerical analysis and plotting respectively

#making array to hold our x and y values
x = np.arange(0,9)
#automatically creating a sequence of numbers from0 to 8 to be stord in the variable x
y = np.array([19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24])
#manually specifying the values to be stored in the variable y

#Calculates the correlation coefficient matrix
#for your variables
correlation = np.corrcoef(x, y)
#extracts the r-value from that matrix
r = correlation[0,1]
#calculated r squared from r
r2 = r * r

print r2
