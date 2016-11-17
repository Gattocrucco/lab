import lab
import scipy.stats as st
from pylab import *

# test for absolute_sigma in fit_linear

#### PARAMETERS ####
m = 1
q = 1
n = 2000
xmean = linspace(0, 1, n)
dx = array([.1] * n)
dy = array([1e-8] * n)
mcn = 1000
####################

# generate mean data
ymean = m * xmean + q
# run fits and collect chisquares
chisq = empty(mcn)
for i in range(mcn):
	# generate data
	deltax = st.norm.rvs(size=n)
	x = xmean + dx * deltax
	deltay = st.norm.rvs(size=n)
	y = ymean + dy * deltay
	# fit
	par, cov = lab.fit_linear(x, y, dx, dy)
	M, Q = par
	chisq[i] = ((y - (M * x + Q))**2 / (dy**2 + (M*dx)**2)).sum()
# plot chisquares
clf()
hist(chisq, bins=int(sqrt(mcn)))
show()
