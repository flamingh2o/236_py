# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 22:58:58 2016

@author: isaacdk
"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
#from scipy import interpolate
import scipy.optimize
#import random

xaxis_label = 'X-axis'
yaxis_label = 'Y-axis'
titleName = 'Semi-random Data'

x_values = np.arange(0, 10, .1)
npts = np.size(x_values)
rand_1 = np.linspace(-0.25, 0.25, 100)
np.random.shuffle(rand_1)
rand_2 = np.linspace(-0.25, 0.25, 100)
np.random.shuffle(rand_2)
y_values = 6.3 * np.e**((-x_values + rand_1)/np.pi) + rand_2

x_min = x_values[0] - 0.1 * np.abs(x_values[0])
x_max = x_values[npts-1] + 0.1 * np.abs(x_values[npts-2])
y_min = np.min(y_values) - 0.1 * np.abs(np.min(y_values))
y_max = np.max(y_values) + 0.1 * np.abs(np.min(y_values))

print('Click on our peak, halfway point, and level off.')
print()
#show plot
plt.figure(1)
plt.clf()
#plt.axis([x_min, x_max, y_min, y_max])
plt.plot(x_values, y_values,'g.')
plt.grid()
plt.xlabel(xaxis_label, fontsize=15)
plt.ylabel(yaxis_label, fontsize=15)
plt.title(titleName, fontsize=20)

#input from user
click = plt.ginput(3, timeout=-1)
x_1 = click[0][0]
x_2 = click[1][0]
x_3 = click[2][0]
y_1 = click[0][1]
y_2 = click[1][1]
y_3 = click[2][1]
amp_g = y_1
tau_g = x_2
x_0_g = x_1
y_0_g = y_3

#define the regions of interest around each peak 
beg = np.int(np.rint(np.interp(x_1, x_values, np.arange(npts))))
end = np.int(np.rint(np.interp(x_3, x_values, np.arange(npts))))
x_roi = x_values[beg:end]
y_roi = y_values[beg:end]

min_x = x_values[beg] - 0.1 * np.abs(x_values[beg])
max_x = x_values[end] + 0.1 * np.abs(x_values[end])
min_y = min(y_values[beg:end]) - 0.1 * np.abs(min(y_values[beg:end]))
max_y = max(y_values[beg:end]) + 0.1 * np.abs(max(y_values[beg:end]))
plt.axis([min_x, max_x, min_y, max_y])

#find a guess for FWHM

#the function itself
def Exp(x_val, a, t, x, y):
    return a * np.e**((-x_val+x)/t) + y
#h = height
#w = fwhm
#b = x val of peak
#c = y val of asymptote

#best fit lines (guesses help the process)
p_guess = [amp_g, tau_g, x_0_g, y_0_g]
peak, pcov = scipy.optimize.curve_fit(Exp, x_roi, y_roi, p0 = p_guess)
perr = np.sqrt(np.diag(pcov))

#plot the fit
plt.plot(x_values, Exp(x_values, *p_guess), 'g--')
plt.plot(x_values, Exp(x_values, *peak), 'r')

print("Our exact fitted values:")
print("Amp :", peak[0])
print("Tau :", peak[1])
print("Center: x =", peak[2])
print("Flatline: y =", peak[3])
print()