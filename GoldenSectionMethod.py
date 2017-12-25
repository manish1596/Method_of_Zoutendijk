
from __future__ import division
from math import sqrt

phi = (1 + sqrt(5))/2

def goldenSectionSearch(function, xLow, xHigh, allowedError) :
	''' Golden Search method to find minimum of funtion in range [xLow, xHigh]
	-> funtion x is assumed to be unimodal in range [xLow, xHigh] '''

	x1 = xHigh - (xHigh - xLow) / phi
	x2 = xLow  + (xHigh - xLow) / phi

	while abs(x2-x1) > allowedError :
		if function(x1) < function(x2) :
			xHigh = x2
		else :
			xLow  = x1

		x1 = xHigh - (xHigh - xLow) / phi
		x2 = xLow  + (xHigh - xLow) / phi		

	return (xLow + xHigh) / 2	

	
