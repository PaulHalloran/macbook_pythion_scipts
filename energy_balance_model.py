import matplotlib.pyplot as plt
import numpy as np


def ebm(alpha,t_in,TSI,emissivity,stephan_bolt_const,transmissivity):
	output_temperature = (1-alpha)*(TSI/4.0)-(emissivity*stephan_bolt_const*np.power(t_in,4)*transmissivity)
	return output_temperature/0.38


alpha = 0.3 # albedo
TSI = 342.0 # energy coming from sun (W/m2)
emissivity = 0.97
transmissivity = 0.64
stephan_bolt_const = 5.670373e-8

t_in = 287.0 

data = []
for i in range(1000):
        data.append(t_in)
	t_in = t_in + ebm(alpha,t_in,TSI,emissivity,stephan_bolt_const,transmissivity)

plt.plot(data)
plt.show()

