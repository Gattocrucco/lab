import lab
import scipy.stats as st
from pylab import *
from scipy.optimize import curve_fit
import time

#### PARAMETERS ####
showplot = True
m = 1 # slope
q = 1 # offset
n = 10000 # number of points
mcn = 1000 # monte carlo runs
fitfun = lab.fit_generic_xyerr4
xmean = linspace(0, 1, n)
dx = array([.1] * n)
dy = array([.1] * n)
# dx = st.norm.rvs(size=n)*.1+.1
# dy = st.norm.rvs(size=n)*.05+.1
# dy = 0
####################

# generate mean data
ymean = m * xmean + q
# run fits and collect chisquares
chisq = empty(mcn)
pars = empty((mcn, 2))
covs = empty((mcn, 2, 2))
times = empty(mcn)
linfun = lambda x, m, q: m * x + q
dxlinfun = lambda x, m, q: array([m] * len(x))
dplinfun = lambda x, m, q: array([x, ones(len(x))])
dpxlinfun = lambda x, m, q: array([ones(len(x)), zeros(len(x))])
ilinfun = lambda y, m, q: y/m - q/m
def dpilinfun(y, m, q):
	rt = empty((2, len(y)))
	rt[0] = (q - y) / m**2 
	rt[1] = -1 / m
	return rt
# dpilinfun = lambda y, m, q: array([-y/m**2, [-1/m]*len(y)])
# def dpilinfun(y, m, q):
# 	rt = empty((2, len(y)))
# 	rt[0] = y
# 	rt[0] /= -m**2
# 	rt[1] = -1/m
# 	return rt
eta = lab.etastart()
for i in range(mcn):
	lab.etastr(eta, i / mcn, mininterval=5)
	# generate data
	deltax = st.norm.rvs(size=n)
	x = xmean + dx * deltax
	deltay = st.norm.rvs(size=n)
	y = ymean + dy * deltay
	# fit
	start = time.time()
	if fitfun == lab.fit_generic_xyerr3:
		par, cov = fitfun(linfun, dxlinfun, dplinfun, dpxlinfun, x, y, dx, dy, (1, 1))
	elif fitfun == lab.fit_linear:
		par, cov = fitfun(x, y, dx, dy)
	elif fitfun == lab.fit_generic_xyerr4:
		par, cov = fitfun(linfun, ilinfun, dplinfun, dpilinfun, x, y, dx, dy, (1, 1))
	elif fitfun == lab.fit_generic_xyerr2:
		par, cov = fitfun(linfun, dxlinfun, dplinfun, x, y, dx, dy, (1, 1))
	end = time.time()
	# save results
	times[i] = end - start
	M, Q = par
	chisq[i] = ((y - (M * x + Q))**2 / (dy**2 + (M*dx)**2)).sum()
	pars[i] = par
	covs[i] = cov

if showplot:
	figure(1)
	clf()
	subplot(421)
	title('$m$')
	hist(pars[:, 0], bins=int(sqrt(mcn)))
	subplot(423)
	title('$q$')
	hist(pars[:, 1], bins=int(sqrt(mcn)))
	subplot(422)
	title('$\Delta m$')
	hist(sqrt(covs[:, 0, 0]), bins=int(sqrt(mcn)))
	subplot(424)
	title('$\Delta q$')
	hist(sqrt(covs[:, 1, 1]), bins=int(sqrt(mcn)))
	subplot(425)
	title('$m,q$')
	plot(pars[:, 0], pars[:, 1], '.k', markersize=2)
	grid()
	subplot(426)
	title('$\\rho mq$')
	hist(covs[:, 0, 1] / sqrt(covs[:, 0, 0]*covs[:, 1, 1]), bins=int(sqrt(mcn)))
	subplot(427)
	title('$\chi^2$')
	hist(chisq, bins=int(sqrt(mcn)))
	subplot(428)
	title('time [ms]')
	hist(times*1000, bins=int(sqrt(mcn)))
	tight_layout()
	show()
