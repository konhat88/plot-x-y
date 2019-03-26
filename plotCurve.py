#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy.optimize import curve_fit

def fit_func(x, a, b):
    return np.exp(a*x + b)

cnt = 1
ws  = ' '
x   = []
y   = []
yn  = []
ymean  = 0
ynmean = 0
ss_tot = 0
ss_err = 0
r2     = 0

# Define the name of the file
f = open('heart.dat')
line = f.readline()

while line:
	line = f.readline()
	splitLine = line.split()
	if len(splitLine) > 1:
		spl = [float(i) for i in splitLine]
		x.append(spl[0])
		y.append(spl[1])
		cnt += 1
#print x, y
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

popt, pcov = curve_fit(fit_func, x, y)


#####################################################################
########               DEFINE THESE                        ##########
#####################################################################

# Define the name and the type of simulation data
plt.plot(x,y,'ro', label='Original', color='black')

# Define the name and the type of the fit curve data
plt.plot(x, fit_func(x, *popt), label='Fitted Curve', color='black')

# Location of the legend
plt.legend(loc='upper right')

# Define axes width
plt.xlim(0,x[cnt-2]+2)
#plt.ylim(0,2)

# Define title of diagram and axes titles
plt.title('thymus dosimetry', fontsize=20)
plt.xlabel('Countries', fontsize=16)
plt.ylabel('Population in million', fontsize=16)

######################################################################


######################################################
#xs = sym.Symbol('\lambda')    
#tex = sym.latex(fit_func(xs,*popt)).replace('$', '')
#plt.title(r'$f(\lambda)= %s$' %(tex),fontsize=16)
######################################################


#print popt
print ('y = exp ( %.3f x + %.3f )' % (popt[0], popt[1]))

for j in range (cnt-2):
	yn.append ( fit_func(x[j], *popt) )
yn = np.array(yn, dtype=float)
#print yn
#print y

for k in range (cnt-2):
	ymean += y[0] / (cnt-1)
	ynmean += yn[0] / (cnt-1)
#print ymean, ynmean

ss_tot = sum ((yi-ymean)**2 for yi in y)
ss_err = sum ((yi-fi)**2 for yi,fi in zip(y,yn))
r2 = 1 - (ss_err/ss_tot)
print ('R2 = %.2f' % (r2))

plt.show()
#plt.savefig('figure.png')



