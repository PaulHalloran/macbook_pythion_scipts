import numpy as np
import matplotlib.pyplot as plt
#loading up modules required to to do
#numerical analysis and plotting respectively

#making array to hold our x and y values
x = np.arange(0,9)
#automatically creating a sequence of numbers from0 to 8 to be stord in the variable x
y = np.array([19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24])
#manually specifying the values to be stored in the variable y

#Calculates the parameters of a least-squares fit
#between x and y. The value '1' is telling it that it is trying to
#fit a stright line. 2 would fit a 2nd order polynomial
#3 a 3rd order polynomial etc. see http://goo.gl/gjSjb8 
parameters = np.polyfit(x, y, 1)
#print the results
print 'Least Squares Linear Regression parameters:'
print parameters

line = parameters[0] * x + parameters[1]
#here we're just calculating the y-values for the
#straight line from the parameters of the linear
#regression (as you learnt earier this term in statistics) 

plt.scatter(x,y)
#produce a scatter plot of the x and y values
plt.plot(x,line,'r')
#add the stright line you've calculated
#in red ('r' - 'k' would be black, 'b' blue, 'y' yellow etc.)
plt.title('Simple Linear Regression')
#add a title to the plot
plt.xlabel('x values')
#add a label to the x-axis
plt.ylabel('y values')
#add a label to the y-axis
plt.show()
#finally display the plot
#or if you want you can use: plt.savefif('my_directory/my_filename.png')
#to save the plot


