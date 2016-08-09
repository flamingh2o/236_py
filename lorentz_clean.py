# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 22:58:58 2016

@author: isaacdk
"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scipy.optimize

#get data file
filename= 'scope2.csv'
xaxis_label = 'Time (s)'
yaxis_label = 'Voltage (V)'
titleName = 'Saturated Absorbtion'

#get data 
#from our scope, the data was on the fourth and fifth columns of the csv
data = np.genfromtxt(filename,delimiter=',')
x_values = data[:,3]
y_values = data[:,4]
npts = np.size(x_values)

print('Click on the beginning and ending of our range--must be from left to right. Then the peak.')
print()
#show plot
plt.figure(1)
plt.clf()
plt.plot(x_values,y_values,'.')
plt.grid()
plt.xlabel(xaxis_label,fontsize=15)
plt.ylabel(yaxis_label,fontsize=15)
plt.title(titleName,fontsize=20)

#input from user
click = plt.ginput(3, timeout=-1)
x_1 = click[0][0]
x_2 = click[1][0]
x_3 = click[2][0]
y_1 = click[0][1]
y_2 = click[1][1]
y_3 = click[2][1]
x_c_g = x_3
y_0_g = (y_1 + y_2)/2
h_g = y_3 - y_0_g
midhght_g = y_0_g + (h_g/2)

#define the regions of interest around each peak 
beg = np.int(np.rint(np.interp(x_1, x_values, np.arange(npts))))
end = np.int(np.rint(np.interp(x_2, x_values, np.arange(npts))))
x_roi = x_values[beg:end]
y_roi = y_values[beg:end]

#find a guess for FWHM
half = np.int(((beg + end)/2))
first_x = x_values[beg:half]
first_y = y_values[beg:half]
f_1 = interpolate.interp1d(first_y, first_x)
sec_x = x_values[half:end]
sec_y = y_values[half:end]
f_2 = interpolate.interp1d(sec_y, sec_x)
w_g = f_2(midhght_g) - f_1(midhght_g)

#the function itself
def Lorentzian(x_val, h, w, x_c, y_0):
    return ((h * w**2)/((w**2)+(4*(x_val - x_c)**2)) + y_0)
#h = height
#w = fwhm
#b = x val of peak
#c = y val of asymptote

#best fit lines (guesses help the process)
p_guess = [h_g, w_g, x_c_g, y_0_g]
peak, pcov = scipy.optimize.curve_fit(Lorentzian, x_roi, y_roi, p0 = p_guess)

perr = np.sqrt(np.diag(pcov))

#plot the fit
plt.plot(x_values, Lorentzian(x_values, *p_guess), 'g')
plt.plot(x_values, Lorentzian(x_values, *peak), 'r')

#the data
#R**2
var = 0
roi_len = end - beg
for x in range(roi_len):
    model_val = Lorentzian(x_roi, *peak)[x]
    y_val = y_roi[x]
    chi2step = (y_val - model_val)**2/(.003**2)
    var = var + chi2step
varr = (1/(roi_len - 4)) * var
print("chi**2: ", varr, sep="")
s_val = 2/(np.sqrt(roi_len-4))
print("s = ", s_val, sep="")
print()

h_f = '%+.4e' % peak[0]
h_e = '%.4e' % perr[0]
h_p = '%.2g' % (np.abs(perr[0]/peak[0]) * 100)
w_f = '%+.4e' % peak[1]
w_e = '%.4e' % perr[1]
w_p = '%.2g' % (np.abs(perr[1]/peak[1]) * 100)
x_c_f = '%+.4e' % peak[2]
x_c_e = '%.4e' % perr[2]
x_c_p = '%.2g' % (np.abs(perr[2]/peak[2]) * 100)
y_0_f = '%+.4e' % peak[3]
y_0_e = '%.4e' % perr[3]
y_0_p = '%.2g' % (np.abs(perr[3]/peak[3]) * 100)

print("Our estimates:")
print("Height:", h_g)
print("FWHM:", w_g)
print("Center: x =", x_c_g)
print("Flatline: y =", y_0_g)
print()

print("Our exact fitted values:")
print("Height:", peak[0])
print("FWHM :", peak[1])
print("Center: x =", peak[2])
print("Flatline: y =", peak[3])
print()

print("Our rounded fits:")
print("Height: ", h_f, " ±", h_e, " (", h_p, "%)", sep="")
print("FWHM: ", w_f, " ±", w_e, " (", w_p, "%)", sep="")
print("Centere: x = ", x_c_f, " ±", x_c_e, " (", x_c_p, "%)", sep="")
print("Flatline: y = ", y_0_f, " ±", y_0_e, " (", y_0_p, "%)", sep="")
print()

print("Fitted equation: (", h_f, " * (", w_f, ")^2) / ( (", w_f, ")^2 + 4*(x - ", x_c_f, ")^2) + ", y_0_f, sep="")