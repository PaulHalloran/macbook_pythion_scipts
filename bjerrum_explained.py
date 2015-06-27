import numpy as np
import carbchem as cc
import matplotlib.pyplot as plt
import re
import string
import itertools

co2 = 280.0
TALK = 2300.0/(1026.*1000.)
TCO2 = 1900.0/(1026.*1000.)
t = 25.0
s = 35.0

def answer(TALK,TCO2,t,s):
    H2CO3 = cc.carbchem(3,-99.0,np.array([t,t]),np.array([s,s]),np.array([TCO2,TCO2]),np.array([TALK,TALK]))[0]
    HCO3 = cc.carbchem(4,-99.0,np.array([t,t]),np.array([s,s]),np.array([TCO2,TCO2]),np.array([TALK,TALK]))[0]
    CO3 = cc.carbchem(5,-99.0,np.array([t,t]),np.array([s,s]),np.array([TCO2,TCO2]),np.array([TALK,TALK]))[0]
    return H2CO3*10000.0,HCO3*10000.0,CO3*10000.0

input = '[AAAA]'
n1 = len(input.split(' ')[0]) - 2
s1 = input[n1+2:]
names = [''.join(x)+s1 for x in itertools.product(string.ascii_lowercase, repeat=n1)]

TALK = 2300.0/(1026.*1000.)
TCO2 = 1900.0/(1026.*1000.)

for i in np.arange(200):
    TCO2 = TCO2+1.0/(1026.*1000.)*5.0
    H2CO3,HCO3,CO3 = answer(TALK,TCO2,t,s)
    fig, ax = plt.subplots()
    val = 2.0
    int = val * 1.2
    x = [0,val]
    y = [H2CO3/val,H2CO3/val]
    ax.plot(x,y,'k')
    ax.fill_between(x,y, 0.0, facecolor='red', alpha=0.65)
    x = [int+0,int+val]
    y = [HCO3/val,HCO3/val]
    ax.plot(x,y,'k')
    ax.fill_between(x,y, 0.0, facecolor='blue', alpha=0.65)
    x = [2*int+0,2*int+val]
    y = [CO3/val,CO3/val]
    ax.plot(x,y,'k')
    ax.fill_between(x,y, 0.0, facecolor='green', alpha=0.65)
    plt.ylim(0,13)
    string = np.str(i)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig('/home/ph290/Desktop/anim/anim_'+names[i]+'.png')
    #plt.show()

TALK = 2000.0/(1026.*1000.)
TCO2 = 2300.0/(1026.*1000.)

for i in np.arange(200):
    TALK = TALK+1.0/(1026.*1000.)*5.0
    H2CO3,HCO3,CO3 = answer(TALK,TCO2,t,s)
    fig, ax = plt.subplots()
    val = 2.0
    int = val * 1.2
    x = [0,val]
    y = [H2CO3/val,H2CO3/val]
    ax.plot(x,y,'k')
    ax.fill_between(x,y, 0.0, facecolor='red', alpha=0.65)
    x = [int+0,int+val]
    y = [HCO3/val,HCO3/val]
    ax.plot(x,y,'k')
    ax.fill_between(x,y, 0.0, facecolor='blue', alpha=0.65)
    x = [2*int+0,2*int+val]
    y = [CO3/val,CO3/val]
    ax.plot(x,y,'k')
    ax.fill_between(x,y, 0.0, facecolor='green', alpha=0.65)
    plt.ylim(0,12)
    string = np.str(i)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig('/home/ph290/Desktop/anim2/anim_'+names[i]+'.png')
