# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:46:31 2021

@author: Ethan
"""

#first we import the various modules and libraries required for our fitting methods
import numpy as np
import scipy.optimize as opt
import scipy.stats as st
import math
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures



# 1. Curve Fit Using Polynomial Regression Method:
    
#defining our polynomial that we will be fitting using polynomial regression
print('\n1. Polynomial to be Fitted with Polynomial Regression:\n')
p = np.poly1d([1, 5, 7, 2])
print(np.poly1d(p))

#creating the values that will be used to plot
x = np.arange(-3, 0, 0.1)
y = p(x)

#plotting the non-fitted polynomial
plt.title('Non-fitted Polynomial')
plt.plot(x, y)
plt.savefig('Non-fitted_Poly1.png')
plt.show()


#defining function for polynomial regression
#the function will plot a polynomial of a selected order to be fitted to 
#our previous polynomial
def pres(order):
    p = np.poly1d(np.polyfit(x, y, order))
    ypred = [p(v) for v in x]
    plt.plot(x, y, 'r', label= 'orig')
    plt.plot(x,ypred,'g', label= 'pred')
    plt.title('fit for order {}'.format(order))
    plt.legend()
    plt.plot()
    plt.savefig('Polynomial_Regression_Fit.png')
#pres function allows the order of the fitted curve to be changed
pres(3)



# 2. Damped Sine Wave Curve Fit:
    
#Defining out function allowing us to call the structure of our sine curve
def fds(xin,sm,sp,ew,gain):
    return [ gain*math.sin(sm*x+sp)*math.exp(-ew*x) for x in xin]

#Setting up points to be plotted and values for our equation
xv = [x/10 for x in range(400)]
yv = fds(xv,4,2,0.1,2)
plt.figure()
plt.title('Damped Sine Wave')
plt.plot(xv,yv)
plt.savefig('Non-fitted_Sine_Wave.png')
plt.show()

#creating the fitted curve that will be plotted with the original curve
popt, pcov = opt.curve_fit(fds,xv,yv)
plt.figure()
plt.plot(xv,yv,'b', label='orig')
plt.plot(xv, fds(xv,*popt), 'g', label='pred')
plt.title('Fit for Damped Sine Wave')
plt.legend()
plt.savefig('Damped_Sine_Wave_Fit.png')
plt.show()



# 4. Curve Fit with Multi-Variables:
    
#we begin by creating two variables
x0 = [x/15 for x in range(100)]
x1 = [math.cos(x/15) for x in range(100)]
xa = (x0, x1)

#function to utilize our two independent variables
def rf(X, fx0, fx1):
    x0, x1=X
    rv= np.sin(np.multiply(x0, fx0)+np.sin(np.multiply(x1,fx1)))
    return rv

#plotting the multivariable curve
yv= rf((x0,x1),2,3)
plt.title('Multi-Variable Curve')
plt.plot(yv)
plt.savefig('Multi-Variable_Curve.png')
plt.show()

#function that allows us to test the multi-variable curve fitting with hints
#as to where to start.
def with_hint(h0, h1):
    popt, pcov = opt.curve_fit(rf, xa, yv, (h0, h1))
    
    plt.title('Multi-Variable Curve Fit: hint ({},{})'.format(h0, h1))
    plt.plot(rf((x0,x1),2,3),'b')
    plt.plot(rf((x0,x1), *popt),'g')
    plt.savefig('Multi-Variable_Curve_Fit.png')
    plt.show()

#First hint (0,0)
with_hint(0, 0)
#This first fit lacks any recognizability of the original curve aside from a 
#faint oscillation akin to a sin/cos wave which this curve does have.

#Second hint (1, 1)
with_hint(1, 1)
#This curve is closer to a proper fit as it is beginning to match some of the
#strange curve shapes that the original curve provides. Though it is far from a match.

#Third hint (100, 100)
with_hint(100, 100)
#this fit is too frequent to match the original, it again shows similarities
#with the original curves inconsistent peak shapes.

#fourth hint (2, 1)
with_hint (2, 1)
#this fit is very close, as the standard sinusoidal shapes of the peaks are
#almost exact in frequency, but is missing some nuances of the actual peaks
#as well as the local max/min

#fifth hint (2, 3)
with_hint(2, 3)
#this is an exact fit to our original curve, including the peaks, as well as 
#local max/min.



# 5. Ridge Regression Method:
    
print('\n\n5. Polynomial to be Fitted with Ridge Regression:\n')
p = np.poly1d([-7, 2, 5, 1])
print(np.poly1d(p))

#creating the values that will be used to plot
x = np.arange(-1, 1, 0.01)
y = p(x)

#plotting the non-fitted polynomial
plt.title('Non-fitted Polynomial')
plt.plot(x, y)
plt.savefig('Non-Fitted_Poly2.png')
plt.show()

#Reshaping the array before creating prediction model and fitting using ridge regression
x= x.reshape(-1, 1)
model = make_pipeline(PolynomialFeatures(4), Ridge())
model.fit(x, y)
y_plot = model.predict(x)

#final plot of the original curve and the fitted curve utilizing ridge regression
plt.title('Polynomial Fit with Ridge Regression')
plt.plot(x,y)
plt.plot(x, y_plot)
plt.savefig('Ridge_Regression_Fit.png')
plt.show()



# 6. Linear Regression Method:
    
print('\n\n6. Polynomial to be Fitted with Linear Regression:\n')
p = np.poly1d([1, -5, 5, 5, -6])
print(np.poly1d(p))

#creating the values that will be used to plot
x = np.arange(-1, 3, 0.1)
y = p(x)

#plotting the non-fitted polynomial
plt.title('Non-fitted Polynomial')
plt.plot(x, y)
plt.savefig('Non-Fitted_Poly3')
plt.show()

#Reshaping the array before creating prediction model and fitting using linear regression
x= x.reshape(-1, 1)
model2 = make_pipeline(PolynomialFeatures(15), LinearRegression())
model2.fit(x, y)
y_plot = model2.predict(x)

#Final plot including original curve and fitted curve using linear regression
plt.title('Polynomial Fit with Linear Regression')
plt.plot(x, y)
plt.plot(x, y_plot)
plt.savefig('Linear_Regression_Fit.png')
plt.show()