import numpy as np

names = ['Julian Brigstocke','Louise Amoore','Dominique Moran','Pat Noxolo','TBC','Sam Kinsley','Paul Harrison','Gordon Walker','Hillary Geoghegan','Isla Forsyth','James Ash','Jemma Wadham','Claire Rambeau','Encarni Montoya','Hendry','Barend van Maanen','McLeod']

length = np.round(len(names)/2.0)

names2 = []

for i in range(np.int(length)):
	print i
	names2.append(names.pop(np.random.randint(0,len(names))))


print names2


